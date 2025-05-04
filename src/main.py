import collections
import itertools
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from graph_transformer_pytorch import GraphTransformer

from data_source import GFNSampler, get_reward_fn_generator, get_smoothed_log_reward, get_uncertain_smoothed_log_reward, get_uniform_counting_log_reward
from gfn import (
    get_loss_to_uniform_backward,
    get_tb_loss_uniform,
    get_tb_loss_adjusted_uniform,
    get_tb_loss_add_node_mult,
    get_tb_loss_const,
    get_tb_loss_rand_const,
    get_tb_loss_rand_var,
    get_tb_loss_aligned,
    get_tb_loss_free,
    get_tb_loss_maxent,
    get_tb_loss_tlm,
    get_tb_loss_biased_tlm,
    get_tb_loss_smooth_tlm,
    get_action_log_probs_test_helper
)


parser = argparse.ArgumentParser()

# general
parser.add_argument("-s", "--seed", type=int, default=1)
parser.add_argument("-d", "--device", type=str, default="cuda", help="generally 'cuda' or 'cpu'")
parser.add_argument("-o", "--save", action="store_true", default=False, help="whether to save outputs to a file")
parser.add_argument("-x", "--test-template", action="store_true", default=False, help="whether to record P_B and P_F embeddings and loss using the results/s/template.npy")
parser.add_argument("-c", "--cycle-len", type=int, default=5, help="how often to log/checkpoint (number of batches)")
parser.add_argument("-t", "--num-test-graphs", type=int, default=64, help="number of graphs to generate for estimating metrics")

# env
parser.add_argument("-b", "--base", type=float, default=0.8, help="base for exponent used in reward calculation")
parser.add_argument("-r", "--reward-idx", type=int, default=0, help="index of reward function to use")

# model
parser.add_argument("-f", "--num-features", type=int, default=10, help="number of features used to represent each node/edge (min 2)")
parser.add_argument("-y", "--depth", type=int, default=1, help="depth of the transformer model")
parser.add_argument("-g", "--max-nodes", type=int, default=8, help="maximum number of nodes in a generated graph")
parser.add_argument("-k", "--max-len", type=int, default=80, help="maximum number of actions per trajectory")
parser.add_argument("-q", "--random-action-template", type=int, default=2, help="index of the random action config to use (see code)")
parser.add_argument("-z", "--log-z", type=float, default=0, help="constant value of log(z) to use (learnt if None)")
parser.add_argument("-i", "--backward_init", type=str, default="random", help="how to initialise the backward policy")

# training
parser.add_argument("-l", "--loss-fn", type=str, default="tb-uniform", help="loss function for training (e.g. TB + uniform backward policy)")
parser.add_argument("-v", "--loss-arg-a", type=float, default=1)
parser.add_argument("-u", "--loss-arg-b", type=float, default=1)
parser.add_argument("-w", "--loss-arg-c", type=float, default=1)
parser.add_argument("-m", "--batch-size", type=int, default=32)
parser.add_argument("-p", "--num-precomputed", type=int, default=16, help="number of trajectories from precomputed, fully connected graphs")
parser.add_argument("-j", "--edges-first", action="store_true", default=False, help="whether to add edges before nodes in precomputed trajectories")
parser.add_argument("-a", "--learning-rate", type=float, default=0.00001)
parser.add_argument("-n", "--max-update-norm", type=float, default=99.9)
parser.add_argument("-e", "--num-batches", type=int, default=5_000)

args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

configs = {
    "tb-uniform": (  # uniform backward policy
        get_tb_loss_uniform,
        {"parameterise_backward": False}
    ),
    "tb-adjusted-uniform": (  # uniform backward policy, with added adjustments to encourage the optimal forward policy
        lambda *pargs, **kwargs: get_tb_loss_adjusted_uniform(*pargs, base=args.base, **kwargs),
        {"parameterise_backward": False}
    ),
    "tb-uniform-add-node": (  # uniform backward policy with an added bias towards adding nodes
        lambda *pargs, **kwargs: get_tb_loss_add_node_mult(*pargs, n=args.loss_arg_a, **kwargs),
        {"parameterise_backward": False}
    ),
    "tb-uniform-const": (  # uniform backward policy with a constant (unnormalised) backward probability
        lambda *pargs, **kwargs: get_tb_loss_const(*pargs, value=args.loss_arg_a, **kwargs),
        {"parameterise_backward": False}
    ),
    "tb-uniform-rand": (  # uniform backward probabilities randomly perturbed by noised sampled from a uniform distribution
        lambda *pargs, **kwargs: get_tb_loss_rand_const(*pargs, mean=args.loss_arg_a, std=args.loss_arg_b, seed=args.loss_arg_c, **kwargs),
        {"parameterise_backward": False}
    ),
    "tb-uniform-rand-var": (  # backward probabilities randomly resampled from a normal distribution on each application
        lambda *pargs, **kwargs: get_tb_loss_rand_var(*pargs, mean=args.loss_arg_a, std=args.loss_arg_b, **kwargs),
        {"parameterise_backward": False}
    ),
    "tb-aligned": (  # aligned to handmade backward policy
        lambda *pargs, **kwargs: get_tb_loss_aligned(*pargs, base=args.base, correct_val=args.loss_arg_a, incorrect_val=args.loss_arg_b, **kwargs),
        {"parameterise_backward": False}
    ),
    "tb-free": (  # TB loss backpropagated to backward policy
        get_tb_loss_free,
        {"parameterise_backward": True}
    ),
    "tb-max-ent": (  # maximum entropy backward policy
        get_tb_loss_maxent,
        {"parameterise_backward": True}
    ),
    "tb-tlm": (  # TLM / pessimistic
        get_tb_loss_tlm,
        {"parameterise_backward": True}
    ),
    "tb-weighted-tlm": (  # TLM / pessimistic with weights toward ns nodes
        lambda *pargs, **kwargs: get_tb_loss_biased_tlm(*pargs, multiplier=args.loss_arg_a, ns=[args.loss_arg_b], **kwargs),
        {"parameterise_backward": True}
    ),
    "tb-smoothed-tlm": (  # TLM / pessimistic mixed with a uniform distribution
        lambda *pargs, **kwargs: get_tb_loss_smooth_tlm(*pargs, a=args.loss_arg_a, **kwargs),  # maybe cosine annealing curve from 1 to 0 over time?
        {"parameterise_backward": True}                                                        # or does it not matter because the backward policy will eventually compensate
    )
}

get_loss, config = configs[args.loss_fn]
parameterise_backward = config["parameterise_backward"]

reward_fns = [get_smoothed_log_reward, get_uncertain_smoothed_log_reward, get_uniform_counting_log_reward]
reward_fn = reward_fns[args.reward_idx]

#compile = lambxa x: torch.compile(x)
compile = lambda x: x

base_model = compile(GraphTransformer(dim=args.num_features, depth=args.depth, edge_dim=args.num_features, with_feedforwards=True, gated_residual=True, rel_pos_emb=False)).to(args.device)

fwd_stop_model = compile(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
fwd_node_model = compile(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
fwd_edge_model = compile(nn.Sequential(nn.Linear(args.num_features*3, args.num_features*3*2), nn.LeakyReLU(), nn.Linear(args.num_features*3*2, 1))).to(args.device)
fwd_models = [fwd_stop_model, fwd_node_model, fwd_edge_model]

if parameterise_backward:
    bck_stop_model = compile(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
    bck_node_model = compile(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
    bck_edge_model = compile(nn.Sequential(nn.Linear(args.num_features*3, args.num_features*3*2), nn.LeakyReLU(), nn.Linear(args.num_features*3*2, 1))).to(args.device)
    bck_models = [bck_stop_model, bck_node_model, bck_edge_model]

    if args.backward_init == "uniform":

        bck_init_optimiser = torch.optim.Adam(itertools.chain(*(i.parameters() for i in bck_models)), lr=args.learning_rate*10, weight_decay=1e-4)
        main_init_optimiser = torch.optim.Adam(base_model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

        # its kind of wasteful that we call fwd_models here even though the actions are random
        init_data_source = GFNSampler(base_model, *fwd_models, lambda nodes, *args, **kwargs: torch.zeros((nodes.shape[0],)),
                                      random_action_prob=1, node_features=args.num_features, edge_features=args.num_features,
                                      max_len=args.max_len, max_nodes=args.max_nodes, batch_size=64, num_precomputed=0, device=args.device)

                                      
        data_loader = torch.utils.data.DataLoader(init_data_source, batch_size=None)
        for it, (jagged_trajs, log_rewards) in zip(range(25), data_loader):

            loss, metrics = get_loss_to_uniform_backward(jagged_trajs, log_rewards, base_model, None, *fwd_models, *bck_models, constant_log_z=1, device=args.device)
            loss.backward()

            main_init_optimiser.step()
            main_init_optimiser.zero_grad()

            bck_init_optimiser.step()
            bck_init_optimiser.zero_grad()

            print(loss.item())

else:
    bck_models = []

if args.loss_fn == "tb-max-ent":
    assert parameterise_backward
    n_model = compile(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
    bck_models.append(n_model)

log_z_model = compile(nn.Linear(1, 1, bias=False)).to(args.device)

random_configs = [(0, 0, 0),
                  (0.5, 0, 0.5),
                  (1, 0.99, 0.1)]
random_prob, random_prob_decay, random_prob_min = random_configs[args.random_action_template]

main_optimiser = torch.optim.Adam(base_model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
fwd_optimiser = torch.optim.Adam(itertools.chain(*(i.parameters() for i in fwd_models)), lr=args.learning_rate*10, weight_decay=1e-4)
log_z_optimiser = torch.optim.Adam(log_z_model.parameters(), lr=args.learning_rate*10, weight_decay=1e-4)

main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(main_optimiser, T_max=args.num_batches)
fwd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fwd_optimiser, T_max=args.num_batches)
log_z_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(log_z_optimiser, T_max=args.num_batches)

if parameterise_backward:
    bck_optimiser = torch.optim.Adam(itertools.chain(*(i.parameters() for i in bck_models)), lr=args.learning_rate*10, weight_decay=1e-4)
    bck_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(bck_optimiser, T_max=args.num_batches)

    # (in case we pretrained the backward policy)
    main_optimiser.zero_grad()
    fwd_optimiser.zero_grad()
    bck_optimiser.zero_grad()
    log_z_optimiser.zero_grad()

reward_fn_generator = get_reward_fn_generator(reward_fn, base=args.base)

data_source = GFNSampler(base_model, *fwd_models, reward_fn_generator,
                         node_features=args.num_features, edge_features=args.num_features,
                         random_action_prob=random_prob, max_len=args.max_len, max_nodes=args.max_nodes, base=args.base,
                         batch_size=args.batch_size, num_precomputed=args.num_precomputed, edges_first=args.edges_first,
                         device=args.device)
data_loader = torch.utils.data.DataLoader(data_source, batch_size=None)

if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    losses = []
    sum_loss = mean_log_reward = mean_connected_prop = mean_num_nodes = 0

    if parameterise_backward:  # TODO: needs updating to adapt to all possible configs
        sum_loss_fwd = sum_loss_bck = 0

    for it, (jagged_trajs, log_rewards) in zip(range(args.num_batches), data_loader):

        loss, metrics = get_loss(jagged_trajs, log_rewards, base_model, log_z_model, *fwd_models, *bck_models, constant_log_z=args.log_z, device=args.device)
        loss.backward()

        params = itertools.chain(*(m.parameters() for m in (base_model, *fwd_models, *bck_models, log_z_model)))
        norm = min(args.max_update_norm, torch.nn.utils.clip_grad_norm_(params, args.max_update_norm).item())

        main_optimiser.step()
        main_optimiser.zero_grad()
        fwd_optimiser.step()
        fwd_optimiser.zero_grad()
        log_z_optimiser.step()
        log_z_optimiser.zero_grad()

        main_scheduler.step()
        fwd_scheduler.step()
        log_z_scheduler.step()

        if parameterise_backward:
            bck_optimiser.step()
            bck_optimiser.zero_grad()
            bck_scheduler.step()

        sum_loss += loss.detach() / args.cycle_len
        mean_log_reward += metrics["mean_log_reward"] / args.cycle_len
        mean_connected_prop += metrics["connected_prop"] / args.cycle_len
        mean_num_nodes += metrics["mean_num_nodes"] / args.cycle_len

        if parameterise_backward:
            sum_loss_fwd += metrics["tb_loss"] / args.cycle_len
            if args.loss_fn not in ["tb-free", "tb-max-ent"]:  # TODO!
                sum_loss_bck += metrics["back_loss"] / args.cycle_len
            else:
                sum_loss_bck += torch.tensor(0)

        with torch.no_grad():

            if (it+1) % 5 == 0:
                data_source.random_action_prob = max(random_prob_min, data_source.random_action_prob * random_prob_decay)

            for m in (base_model, log_z_model, *fwd_models, *bck_models):
                m.eval()

            # TODO: temp
            #if (it+1)%args.cycle_len == 0:
            if it+1 in [1, 500, 1_000, 2_000, 5_000, 10_000]:

                test_mean_log_reward = 0
                test_node_counts, test_connectivities = [], []

                graphs = data_source.generate_graphs(args.num_test_graphs)
                for i, (nodes, edges, masks) in enumerate(graphs):
                    num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
                    num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))

                    test_mean_log_reward += reward_fn(nodes.reshape((1, *nodes.shape)), edges.reshape((1, *edges.shape)), alpha=1_000_000).item()
                    test_connectivities.append(int(num_edges == num_nodes**2))
                    test_node_counts.append(num_nodes.item())

                    #if args.save:
                    #    np.save(f"results/batches/nodes_{it}_{i}.npy", nodes.to("cpu").numpy())
                    #    np.save(f"results/batches/edges_{it}_{i}.npy", edges.to("cpu").numpy())
                    #    np.save(f"results/batches/masks_{it}_{i}.npy", edges.to("cpu").numpy())

                test_mean_log_reward /= args.num_test_graphs if args.num_test_graphs != 0 else 1
                test_mean_connected_prop = sum(test_connectivities) / max(len(test_connectivities), 1)

                test_node_count_distribution = collections.Counter(test_node_counts)

                ens_0 = data_source.get_log_unnormalised_ens()
                ens_1 = data_source.get_log_unnormalised_ens(refl=True)


                # assume that samples are uniformly generated in these buckets (questionalble)
                # does this make it a lower bound?
                gen_distribution = np.array([0 for __ in range(1, 9) for c in ["d", "c"]], dtype=float)
                for n, c in zip(test_node_counts, test_connectivities):
                    gen_distribution[2*(n-1) + c] += 1
                gen_distribution /= len(test_connectivities)

                if args.reward_idx == 2:
                    tru_distribution = np.array([v for n in range(1, 9) for v in [(1 - 2 ** (- n ** 2)) / 8, (2 ** (- n ** 2)) / 8]])
                else:
                    s = args.base*(1-args.base**8)/(1-args.base)
                    tru_distribution = np.array([v for n in range(1, 9) for v in [0, (args.base ** n)/s]])

                eta = 0.001
                # https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
                ks = np.max(np.abs(np.cumsum(gen_distribution) - np.cumsum(tru_distribution)))
                # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence (KL(true || gen))
                kl = np.sum(np.maximum(eta, tru_distribution) * np.log(np.maximum(eta, tru_distribution) / np.maximum(eta, gen_distribution)))
                # https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence (questionable usefulness here?)
                m_distribution = (tru_distribution + gen_distribution) / 2
                js = (np.sum(np.maximum(eta, tru_distribution) * np.log(np.maximum(eta, tru_distribution) / np.maximum(eta, m_distribution))) \
                    + np.sum(np.maximum(eta, gen_distribution) * np.log(np.maximum(eta, gen_distribution) / np.maximum(eta, m_distribution)))) / 2

                print(
                    f"{it: <5} loss: {sum_loss.item():7.2f}" \
                      + (f" (fwd: {sum_loss_fwd.item():7.2f}, bck: {sum_loss_bck.item():7.2f})" if parameterise_backward else "") + \
                    f"; norm: {norm:6.3f}; " \
                    f"log(z): {metrics['log_z']:6.3f}; " \
                    f"mean log reward: {test_mean_log_reward:8.3f} ({mean_log_reward:8.3f}); " \
                    f"connected: {test_mean_connected_prop:4.2f} ({mean_connected_prop:4.2f}); " \
                    f"ens_0: [{', '.join([f'{i.item():6.2f}' for i in ens_0])}]; " \
                    f"ens_1: [{', '.join([f'{i.item():6.2f}' for i in ens_1[1:]])}]; " \
                    f"({mean_num_nodes:3.1f}; {len(graphs)}), {', '.join([f'{i}: {test_node_count_distribution[i]:0>2}' for i in range(1, 9)])}; " \
                    f"ks: {ks:8.5f}; kl: {kl:8.5f}; js: {js:8.5f}"  # TODO: what the hell
                )

                if args.test_template:  # TODO: temp (integrate this in with normal running)

                    total_loss = 0
                    for i in range(8):  # TODO: set to 32 for 3d plot sweep
                        jagged_trajs, log_rewards = next(data_source)
                        loss, metrics = get_loss(
                            jagged_trajs, log_rewards, base_model, log_z_model, *fwd_models, *bck_models, constant_log_z=args.log_z, device=args.device
                        )
                        total_loss += metrics["tb_loss"]

                        if i < 8:
                            np.save(f"results/batches/trajs_{it}_{i}.npy", np.array(jagged_trajs, dtype=object), allow_pickle=True)

                    losses.append(total_loss.item())
                    np.save(f"results/losses.npy", np.array(losses))

                    for nodes, edges, masks in np.load("results/s/template.npy", allow_pickle=True):
                        fwd_action_probs, bck_action_probs = get_action_log_probs_test_helper(nodes, edges, masks, args.loss_fn,
                                                                                              base_model, fwd_models, bck_models, log_z_model,
                                                                                              args.log_z, args.loss_arg_a, args.loss_arg_b, args.loss_arg_c,
                                                                                              device=args.device)
                        np.save(f"results/embeddings/fwd_{it}.npy", torch.flatten(fwd_action_probs).to("cpu").numpy())
                        if bck_action_probs is not None:
                            np.save(f"results/embeddings/bck_{it}.npy", torch.flatten(bck_action_probs).to("cpu").numpy())

                if args.save:
                    names = ("stop_model", "node_model", "edge_model")
                    for m, f in zip([base_model, *fwd_models, *bck_models, log_z_model],
                                    ["base_model", *("fwd_" + n for n in names), *("bck_" + n for n in names), "log_z_model"]):
                        torch.save(m.state_dict(), f"results/models/{f}_{it}.pt")

                    np.save(f"results/metrics/{it}", np.array([
                        it,
                        sum_loss.item(),
                        *([
                            sum_loss_fwd.item(),
                            sum_loss_bck.item()
                        ] if parameterise_backward else []),
                        norm,
                        metrics["log_z"],
                        test_mean_log_reward,
                        mean_log_reward,
                        test_mean_connected_prop,
                        mean_connected_prop,
                        mean_num_nodes,
                        len(graphs),
                        *[test_node_count_distribution[i] for i in range(1, 9)],
                        ks, kl, js
                    ]))

                sum_loss = mean_log_z = mean_log_reward = mean_connected_prop = mean_num_nodes = 0

                if parameterise_backward:  # needs updating to adapt to all possible configs
                    sum_loss_fwd = sum_loss_bck = 0

    print("done.")
