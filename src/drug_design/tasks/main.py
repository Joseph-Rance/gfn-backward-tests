import copy
from itertools import cycle
import argparse
import pickle
import os
from shutil import rmtree
import time
import numpy as np
from scipy.spatial.distance import pdist
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem

from synflownet.envs.synthesis_building_env import ReactionTemplateEnvContext

from drug_design.tasks.n_model import GraphTransformerSynGFN
from drug_design.tasks.algs import (TrajectoryBalanceUniform,
                                    TrajectoryBalanceFree,
                                    TrajectoryBalanceTLM,
                                    TrajectoryBalanceMaxEnt,
                                    TrajectoryBalancePrefDQN,
                                    TrajectoryBalancePrefREINFORCE,
                                    TrajectoryBalancePrefAC,
                                    TrajectoryBalancePrefPPO,
                                    sq_dist, huber)
from drug_design.tasks.util import ReactionTask, ReactionTemplateEnv, SynthesisSampler, DataSource


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", type=str, default="cuda")
parser.add_argument("-b", "--batch-size", type=int, default=64)
parser.add_argument("-i", "--config-idx", type=int, default=0, help="index of config to select (see main.py)")
parser.add_argument("-w", "--preference-strength", type=float, default=0, help="cost preference weighting in reward")
parser.add_argument("-g", "--gamma", type=float, default=1, help="discount factor on rewards for backward policy")
parser.add_argument("-m", "--entropy-loss-multiplier", type=float, default=0, help="weighting of entropy term in backward policy loss")
parser.add_argument("-e", "--epsilon", type=float, default=0.2, help="clip proportion for PPO")
parser.add_argument("-l", "--dist-fn", type=str, default="huber", help="options: {square, huber}")
parser.add_argument("-f", "--target-update-period", type=int, default=5, help="number of batches between each update to the target model")
parser.add_argument("-p", "--print-period", type=int, default=1, help="number of batches between each print")
parser.add_argument("-c", "--checkpoint-period", type=int, default=5, help="number of batches between each checkpoint")
parser.add_argument("-f", "--reward-thresh", type=float, default=0.9, help="value required to be considered 'high' reward")
args = parser.parse_args()


# UNIFORM BACKWARD POLICY
config = {
    "preference_strength": args.preference_strength,  # cost preference weighting in reward
    "algo": TrajectoryBalanceUniform,  # algorithm to determine policy loss
    "parameterise_p_b": False,  # whether to learn P_b
    "sample_backward": False,  # whether to sample trajectories using the backward policy
    "target_model": False,  # whether to maintain a backward model
    "gamma": None,  # discount factor on rewards for backward policy
    "entropy_loss_multiplier": None,  # weighting of entropy term in backward policy loss
    "target_update_period": None,  # frequency to update target model with main weights
    "epsilon": None,  # clip proportion for PPO
    "outs": 1  # number of per graph outputs
}

if args.c == 1:  # UNCONSTRAINED BACKWARD POLICY
    config["algo"] = TrajectoryBalanceFree
    config["parameterise_p_b"] = True
elif args.c == 2:  # TRAJECTORY LIKELIHOOD MAXIMISATION BACKWARD POLICY
    config["algo"] = TrajectoryBalanceTLM
    config["parameterise_p_b"] = True
elif args.c == 3:  # MAXIMUM ENTROPY BACKWARD POLICY
    config["algo"] = TrajectoryBalanceMaxEnt
    config["parameterise_p_b"] = True
    config["outs"] = 2
elif args.c == 4:  # PREFERENCE BACKWARD POLICY WITH DQN
    config["algo"] = TrajectoryBalancePrefDQN
    config["parameterise_p_b"] = config["sample_backward"] = config["target_model"] = True
    config["gamma"] = args.gamma
    config["entropy_loss_multiplier"] = args.entropy_loss_multiplier
    config["target_update_period"] = args.target_update_period
elif args.c == 5:  # PREFERENCE BACKWARD POLICY WITH REINFORCE
    config["algo"] = TrajectoryBalancePrefREINFORCE
    config["parameterise_p_b"] = config["sample_backward"] = True
    config["gamma"] = args.gamma
    config["entropy_loss_multiplier"] = args.entropy_loss_multiplier
    config["outs"] = 2
elif args.c == 6:  # PREFERENCE BACKWARD POLICY WITH ACTOR CRITIC
    config["algo"] = TrajectoryBalancePrefAC
    config["parameterise_p_b"] = config["sample_backward"] = True
    config["gamma"] = args.gamma
    config["entropy_loss_multiplier"] = args.entropy_loss_multiplier
    config["outs"] = 2
elif args.c == 7:  # PREFERENCE BACKWARD POLICY WITH PPO
    config["algo"] = TrajectoryBalancePrefPPO
    config["parameterise_p_b"] = config["sample_backward"] = config["target_model"] = True
    config["gamma"] = args.gamma
    config["entropy_loss_multiplier"] = args.entropy_loss_multiplier
    config["target_update_period"] = args.target_update_period
    config["epsilon"] = args.epsilon
    config["outs"] = 2


if __name__ == "__main__":

    rmtree("results", ignore_errors=True)
    os.mkdir("results")
    os.mkdir("results/models")
    os.mkdir("results/batches")

    rel_path = "/".join(os.path.abspath(__file__).split("/")[:-2])

    with open(rel_path + "/data/building_blocks/enamine_bbs.txt", "r") as file:
        building_blocks = file.read().splitlines()

    with open(rel_path + "/data/templates/hb.txt", "r") as file:
        reaction_templates = file.read().splitlines()

    with open(rel_path + "/data/building_blocks/precomputed_bb_masks_enamine_bbs.pkl", "rb") as f:
        precomputed_bb_masks = pickle.load(f)

    task = ReactionTask(args.device)  # for reward
    ctx = ReactionTemplateEnvContext(  # deals with molecules
        num_cond_dim=task.num_cond_dim,
        building_blocks=building_blocks,
        reaction_templates=reaction_templates,
        precomputed_bb_masks=precomputed_bb_masks,
        fp_type="morgan_1024",
        fp_path=None,
        strict_bck_masking=False
    )
    env = ReactionTemplateEnv(ctx)  # for actions
    sampler = SynthesisSampler(ctx, env, args.device, config["parameterise_p_b"], not config["parameterise_p_b"])  # for sampling policies
    algo = config["algo"](ctx, sampler, args.device, config["preference_strength"],
                gamma=config["gamma"], entropy_loss_multiplier=config["entropy_loss_multiplier"], eps=config["epsilon"])  # for computing loss / helps making batches
    model = GraphTransformerSynGFN(ctx, do_bck=config["parameterise_p_b"], outs=config["outs"])
    target_model = GraphTransformerSynGFN(ctx, do_bck=config["parameterise_p_b"], outs=config["outs"]) if config["target_model"] else None

    Z_params = list(model._logZ.parameters())
    non_Z_params = [i for i in model.parameters() if all(id(i) != id(j) for j in Z_params)]

    dist_fn = sq_dist if args.dist_fn == "square" else huber

    opt = torch.optim.Adam(non_Z_params, 1e-4, (0.9, 0.999), weight_decay=1e-8, eps=1e-8)
    opt_Z = torch.optim.Adam(Z_params, 1e-3, (0.9, 0.999), weight_decay=1e-8, eps=1e-8)

    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda steps: 2 ** -(steps/200))
    lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(opt_Z, lambda steps: 2 ** -(steps/5_000))

    model.to(args.device)

    with torch.no_grad():  # (probably not necessary)
        train_src = DataSource(ctx, algo, task, args.device, use_replay=config["sample_backward"])  # gets training data inc rewards

        train_src.do_sample_model(model, args.batch_size)
        if config["sample_backward"]:
            train_src.do_sample_backward(model, args.batch_size)

        train_dl = torch.utils.data.DataLoader(train_src, batch_size=None)

    unique_scaffolds = set()
    num_unique_scaffolds = [0]
    num_mols_tested = [0]

    mean_tanimoto_distances = []
    traj_len_dist = [0 for __ in range(7)]

    full_results = [[] for __ in range(20)]

    start_time = time.time()

    for it, batch in zip(range(5_000), cycle(train_dl)):

        batch = batch.to(args.device)

        if config["target_model"] and (it+1) % config["target_update_period"] == 0:
            # copy probably redundant here
            target_model.load_state_dict(copy.deepcopy(model.state_dict()))

        model.train()

        loss, info = algo.compute_batch_losses(model, batch, target_model, dist_fn=dist_fn)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        opt.step()
        opt.zero_grad()
        opt_Z.step()
        opt_Z.zero_grad()

        lr_sched.step()
        lr_sched_Z.step()

        for l in batch.traj_lens:
            traj_len_dist[l] += 1 / 5_000 / args.batch_size

        # test number of unique high reward scaffolds we found *during training*
        with torch.no_grad():

            mols = [ctx.graph_to_obj(batch.nx_graphs[i]) for i in (torch.cumsum(batch.traj_lens, 0) - 1)]
            rewards = torch.exp(batch.log_rewards / batch.cond_info_beta)

            murcko_scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]

            scaffolds_above_thresh = [smi for smi, r in zip(murcko_scaffolds, rewards) if r > args.reward_thresh]
            unique_scaffolds.update(scaffolds_above_thresh)

            num_mols_tested.append(num_mols_tested[-1] + len(mols))
            num_unique_scaffolds.append(len(unique_scaffolds))

            np.save("results/unique_scaffolds.npy", list(zip(num_mols_tested, num_unique_scaffolds)))

            fps = np.array([np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2000)) for mol in mols])
            mean_tanimoto_dist = np.mean(pdist(fps, metric="jaccard"))
            mean_tanimoto_distances.append(mean_tanimoto_dist)

            np.save("results/tanimoto_distances.npy", mean_tanimoto_distances)

            total_time = time.time() - start_time
            start_time = time.time()

            loss_0 = info.get("tb_loss", 0)
            loss_1 = info.get("n_loss", 0) + info.get("p_b_loss", 0) + info.get("back_loss", 0)
            loss_2 = info.get("critic_loss", 0) + info.get("baseline_loss", 0)

            if (it+1) % args.print_period == 0:
                print(f"iteration {it} : loss:{info['loss']:7.3f} ({' + '.join((f'{l:7.3f}' for l in (loss_0, loss_1, loss_2)))})" \
                    f"sampled_reward_avg:{rewards.mean().item():6.4f} " \
                    f"time_spent:{total_time:4.2f} " \
                    f"logZ:{info['log_z']:7.4f} " \
                    f"gen scaffolds: {len(scaffolds_above_thresh)} " \
                    f"unique scaffolds: {len(unique_scaffolds)} " \
                    f"Tanimoto: {mean_tanimoto_dist} " \
                    f"mean synth. cost: {sum([i.sum().item() for i in batch.bbs_costs])/len(batch.traj_lens):4.2f}")

            res = [
                info["loss"],
                loss_0,
                loss_1,
                loss_2,
                rewards.mean().item(),
                total_time,
                info["log_z"],
                len(scaffolds_above_thresh),
                len(unique_scaffolds),
                sum([i.sum().item() for i in batch.bbs_costs]) / len(batch.traj_lens),
                info["log_p_f"],
                info["log_p_b"],
                info["bck_std"],
                *traj_len_dist
            ]

            for idx, r in enumerate(res):
                full_results[idx].append(r)

            np.save("results/results.npy", np.array(full_results))

            if (it+1) % args.checkpoint_period == 0:
                torch.save(model.state_dict(), f"results/models/{it}.pt")
                np.save(f"results/batches/{it}.npy", np.array([batch.cpu()], dtype=object), allow_pickle=True)
