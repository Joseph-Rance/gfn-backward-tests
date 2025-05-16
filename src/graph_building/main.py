import math
import collections
import itertools
import argparse
import random
import time
import pickle
import sys
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from graph_transformer_pytorch import GraphTransformer

from data_source import (
    GFNSampler,
    get_reward_fn_generator,
    get_smoothed_overfit_log_reward,
    get_uniform_counting_log_reward,
    get_cliques_log_reward,
    uniform_true_dist,
    overfit_true_dist,
    cliques_true_dist
)
from gfn import get_get_tb_loss_backward, get_loss_to_uniform_backward, get_metrics, trajs_to_tensors 
from backward import (
    const,
    uniform,
    action_mult,
    rand,
    aligned,
    frozen,
    free,
    tlm,
    soft_tlm,
    smooth_tlm,
    biased_tlm,
    max_ent,
    loss_aligned,
    meta
)
from util import get_graphs_above_threshold

with open("results/experiment_config.sh", "w") as f:
    f.write(" ".join(sys.argv))

parser = argparse.ArgumentParser()

# general
parser.add_argument("-s", "--seed", type=int, default=1)
parser.add_argument("-d", "--device", type=str, default="cuda", help="generally 'cuda' or 'cpu'")
parser.add_argument("-o", "--save", action="store_true", default=False, help="whether to save outputs to a file")
parser.add_argument("-c", "--cycle-len", type=int, default=100, help="how often to log/checkpoint (number of batches)")
parser.add_argument("-t", "--num-test-graphs", type=int, default=1024, help="number of graphs to generate for estimating metrics")

# env
parser.add_argument("-r", "--reward-idx", type=int, default=2, help="index of reward function to use")
parser.add_argument("-e", "--reward-arg", type=float, default=[0, 0.8, 3], help="base for exponent used in reward calculation; size of cliques to search for")

# model
parser.add_argument("-f", "--num-features", type=int, default=16, help="number of features used to represent each node/edge (min 2)")
parser.add_argument("-y", "--depth", type=int, default=2, help="depth of the transformer model")
parser.add_argument("-g", "--max-nodes", type=int, default=9, help="maximum number of nodes in a generated graph")
parser.add_argument("-q", "--random-action-template", type=int, default=2, help="index of the random action config to use (see code)")
parser.add_argument("-z", "--log-z", type=float, default=None, help="constant value of log(z) to use (learnt if None)")
parser.add_argument("-i", "--backward-init", type=str, default="random", help="how to initialise the backward policy; one of {random, uniform, <directory name>}")
parser.add_argument("-j", "--history-bounds", type=int, default=1, help="controls how much the MDP is like a tree")

# training
parser.add_argument("-l", "--loss-fn", type=str, default="tb-uniform", help="loss function for training (e.g. TB + uniform backward policy)")
parser.add_argument("-v", "--loss-arg-a", type=float, default=1)
parser.add_argument("-u", "--loss-arg-b", type=float, default=1)
parser.add_argument("-w", "--loss-arg-c", type=float, default=1)
parser.add_argument("-b", "--batch-size", type=int, default=128)
parser.add_argument("-p", "--num-precomputed", type=int, default=0, help="number of trajectories from precomputed, high-reward graphs")
parser.add_argument("-a", "--learning-rate", type=float, default=0.0005)
parser.add_argument("-n", "--max-update-norm", type=float, default=499.9)
parser.add_argument("-k", "--num-batches", type=int, default=10_000)
parser.add_argument("-x", "--backward-reset-period", type=int, default=-1, help="how often to reset the backward policy (-1 for no resets)")
parser.add_argument("-m", "--meta-test", action="store_true", default=False, help="whether to save outputs for meta learning")

args = parser.parse_args()

NO_TEMPLATE = True  # for if there is no template yet (need to run this file before generating template)
args.reward_arg = args.reward_arg[args.reward_idx]  # this is a bit hacky but better than forgetting to update this argument

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

configs = {
    "tb-const": (const, {"parameterise_backward": False, "args": {"value": args.loss_arg_a}}),  # uniform backward policy with a constant (unnormalised) backward probability
    "tb-uniform": (uniform, {"parameterise_backward": False, "args": {}}),  # uniform backward policy
    "tb-uniform-action-mult": (action_mult, {"parameterise_backward": False, "args": {"action_type": args.loss_arg_a, "n": args.loss_arg_b}}),  # uniform backward with bias towards action_type
    "tb-uniform-rand": (rand, {"parameterise_backward": False, "args": {"std": args.loss_arg_a}}),  # backward probabilities resampled from a normal distribution on each application
    "tb-aligned": (aligned, {"parameterise_backward": False, "args": {"reward_arg": args.reward_arg, "cor_val": args.loss_arg_a, "inc_val": args.loss_arg_b}}),  # align handmade backward policy
    "tb-frozen": (frozen, {"parameterise_backward": True, "args": {}}),  # frozen parameterised backward policy
    "tb-free": (free, {"parameterise_backward": True, "args": {}}),  # TB loss backpropagated to backward policy
    "tb-tlm": (tlm, {"parameterise_backward": True, "args": {}}),  # TLM / pessimistic
    "tb-soft-tlm": (soft_tlm, {"parameterise_backward": True, "args": {"a": args.loss_arg_a}}),  # TLM / pessimistic mixed with a uniform distribution
    "tb-smooth-tlm": (smooth_tlm, {"parameterise_backward": True, "args": {"a": args.loss_arg_a}}),  # TLM / pessimistic mixed with a uniform distribution (pre-backprop)
    "tb-biased-tlm": (biased_tlm, {"parameterise_backward": True, "args": {"multiplier": args.loss_arg_a, "ns": [args.loss_arg_b]}}),  # TLM / pessimistic with weights toward ns nodes
    "tb-max-ent": (max_ent, {"parameterise_backward": True, "args": {}}),  # maximum entropy backward policy
    "tb-loss-aligned": (loss_aligned, {"parameterise_backward": False, "args": {"iters": args.loss_arg_a, "std_mult": args.loss_arg_b}}),  # aligned to loss-based backward policy
    "meta": (meta, {"parameterise_backward": False, "args": {"weights": torch.load("results/meta_weights.pt") if args.meta_test else None, "reward_arg": args.reward_arg}})  # for meta learning
}

backward, config = configs[args.loss_fn]
get_loss = lambda *pargs, **kwargs: get_get_tb_loss_backward(*backward)(*pargs, **config["args"], **kwargs)
parameterise_backward = config["parameterise_backward"]

reward_fns = [get_uniform_counting_log_reward, get_smoothed_overfit_log_reward, get_cliques_log_reward]
reward_fn = reward_fns[args.reward_idx]
high_reward_threshold = (-10, -0.8, 3)[args.reward_idx]
tru_distribution = (uniform_true_dist, overfit_true_dist, cliques_true_dist)[args.reward_idx]

#compile_model = lambxa x: torch.compile(x)
compile_model = lambda x: x

base_models = [compile_model(GraphTransformer(dim=args.num_features, depth=args.depth, edge_dim=args.num_features, with_feedforwards=True, gated_residual=True, rel_pos_emb=False)).to(args.device)]

fwd_stop_model = compile_model(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
fwd_node_model = compile_model(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
fwd_edge_model = compile_model(nn.Sequential(nn.Linear(args.num_features*3, args.num_features*3*2), nn.LeakyReLU(), nn.Linear(args.num_features*3*2, 1))).to(args.device)
fwd_models = [fwd_stop_model, fwd_node_model, fwd_edge_model]

if parameterise_backward:
    bck_stop_model = compile_model(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
    bck_node_model = compile_model(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
    bck_edge_model = compile_model(nn.Sequential(nn.Linear(args.num_features*3, args.num_features*3*2), nn.LeakyReLU(), nn.Linear(args.num_features*3*2, 1))).to(args.device)
    bck_models = [bck_stop_model, bck_node_model, bck_edge_model]

    if args.backward_init == "uniform":

        bck_init_optimiser = torch.optim.Adam(itertools.chain(*(i.parameters() for i in bck_models)), lr=args.learning_rate*10, weight_decay=1e-4)
        main_init_optimiser = torch.optim.Adam(base_models[0].parameters(), lr=args.learning_rate, weight_decay=1e-4)

        # its kind of wasteful that we call fwd_models here even though the actions are random
        init_data_source = GFNSampler(base_models[0], *fwd_models, lambda nodes, *args, **kwargs: torch.zeros((nodes.shape[0],)),
                                      random_action_prob=1, node_features=args.num_features, edge_features=args.num_features,
                                      max_len=args.max_len, max_nodes=args.max_nodes, batch_size=64, num_precomputed=0, device=args.device)

                                      
        data_loader = torch.utils.data.DataLoader(init_data_source, batch_size=None)
        for it, (jagged_trajs, log_rewards) in zip(range(25), data_loader):

            loss, metrics = get_loss_to_uniform_backward(jagged_trajs, log_rewards, base_models[0], None, *fwd_models, *bck_models, constant_log_z=1, device=args.device)
            loss.backward()

            main_init_optimiser.step()
            main_init_optimiser.zero_grad()

            bck_init_optimiser.step()
            bck_init_optimiser.zero_grad()

    elif args.backward_init != "random":  # args.backward_init should be the directory containing the initial backward policy

        base_models.append(compile_model(GraphTransformer(dim=args.num_features, depth=args.depth, edge_dim=args.num_features, with_feedforwards=True, gated_residual=True, rel_pos_emb=False)).to(args.device))
        base_models[-1].load_state_dict(torch.load(f"{args.backward_init}/base_model.pt", weights_only=True))
        bck_stop_model.load_state_dict(torch.load(f"{args.backward_init}/bck_stop_model.pt", weights_only=True))
        bck_node_model.load_state_dict(torch.load(f"{args.backward_init}/bck_node_model.pt", weights_only=True))
        bck_edge_model.load_state_dict(torch.load(f"{args.backward_init}/bck_edge_model.pt", weights_only=True))

else:
    bck_models = []

if args.loss_fn == "tb-max-ent":
    n_model = compile_model(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
    bck_models.append(n_model)

log_z_model = compile_model(nn.Linear(1, 1, bias=False)).to(args.device)

random_configs = [(0, 0, 0),
                  (0.5, 0, 0.5),
                  (0.8, 0.99, 0.15),
                  (0.8, 0.99, 0)]
random_prob, random_prob_decay, random_prob_min = random_configs[args.random_action_template]

main_optimiser = torch.optim.Adam(base_models[0].parameters(), lr=args.learning_rate, weight_decay=1e-6)
fwd_optimiser = torch.optim.Adam(itertools.chain(*(i.parameters() for i in fwd_models)), lr=args.learning_rate, weight_decay=1e-6)
log_z_optimiser = torch.optim.Adam(log_z_model.parameters(), lr=args.learning_rate, weight_decay=1e-6)

#main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(main_optimiser, patience=1_000)
#fwd_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fwd_optimiser, patience=1_000)
#log_z_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(log_z_optimiser, patience=1_000)

if parameterise_backward:
    bck_optimiser = torch.optim.Adam(itertools.chain(*(i.parameters() for i in bck_models)), lr=args.learning_rate*10, weight_decay=1e-4)
    #bck_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(bck_optimiser, patience=1_000)

    # in case we pretrained the backward policy
    main_optimiser.zero_grad()
    fwd_optimiser.zero_grad()
    bck_optimiser.zero_grad()
    log_z_optimiser.zero_grad()

reward_fn_generator = get_reward_fn_generator(reward_fn, reward_arg=args.reward_arg)

data_source = GFNSampler(base_models[0], *fwd_models, reward_fn_generator,
                         node_features=args.num_features, edge_features=args.num_features,
                         random_action_prob=random_prob, max_len=args.max_len, max_nodes=args.max_nodes, reward_arg=args.reward_arg,
                         node_history_bounds=(0, args.history_bounds), edge_history_bounds=(0, args.history_bounds),
                         batch_size=args.batch_size, num_precomputed=args.num_precomputed, edges_first=False,
                         device=args.device)
data_loader = torch.utils.data.DataLoader(data_source, batch_size=None)

if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    train_metrics = {}
    train_time = test_time = 0
    graphs_above_threshold, prev_graphs_above_threshold = set(), 0

    for it, (jagged_trajs, log_rewards) in zip(range(args.num_batches), data_loader):

        train_time -= time.perf_counter()

        loss, curr_metrics = get_loss(jagged_trajs, log_rewards, base_models, log_z_model, *fwd_models, *bck_models, constant_log_z=args.log_z, device=args.device)
        loss.backward()

        params = itertools.chain(*(m.parameters() for m in (*base_models, *fwd_models, *bck_models, log_z_model)))
        norm = min(args.max_update_norm, torch.nn.utils.clip_grad_norm_(params, args.max_update_norm).item())

        main_optimiser.step()
        main_optimiser.zero_grad()
        fwd_optimiser.step()
        fwd_optimiser.zero_grad()
        log_z_optimiser.step()
        log_z_optimiser.zero_grad()

        #main_scheduler.step()
        #fwd_scheduler.step()
        #log_z_scheduler.step()

        if parameterise_backward:
            bck_optimiser.step()
            bck_optimiser.zero_grad()
            #bck_scheduler.step()

        graphs_above_threshold.update(get_graphs_above_threshold(jagged_trajs, log_rewards, threshold=high_reward_threshold))  # need to set this threshold correctly!

        for k,v in curr_metrics.items():
            train_metrics[f"train_{k}"] = train_metrics.get(f"train_{k}", 0) + v / args.cycle_len

        train_metrics["train_norm"] = train_metrics.get("train_norm", 0) + norm / args.cycle_len

        with torch.no_grad():

            train_time += time.perf_counter()

            if args.cycle_len > 0 and (it+1)%args.cycle_len == 0:

                test_time -= time.perf_counter()

                for m in (*base_models, log_z_model, *fwd_models, *bck_models):
                    m.eval()

                train_metrics["graphs_above_threshold"] = len(graphs_above_threshold)
                train_metrics["new_graphs_above_threshold"] = prev_graphs_above_threshold - len(graphs_above_threshold)
                prev_graphs_above_threshold = len(graphs_above_threshold)

                template_metrics = {}

                fwd_embs, bck_embs = []
                template = np.load("results/s/template.npy", allow_pickle=True)
                for nodes, edges, masks, actions, traj_lens in template:

                    log_rewards = []
                    for graph_idx in (traj_lens - 2):
                        log_rewards.append(reward_fn(nodes[graph_idx], edges[graph_idx]))
                    log_rewards = torch.tensor(log_rewards)

                    curr_metrics, fwd_action_probs, bck_action_probs = get_metrics(nodes, edges, masks, actions, traj_lens, log_rewards,
                                                                                   base_models, fwd_models, bck_models, args.log_z, log_z_model,
                                                                                   *backward, **config["args"], device=args.device)
                    for k,v in curr_metrics.items():
                        template_metrics[f"template_{k}"] = template_metrics.get(f"template_{k}", 0) + v / len(template)  # (could save some divides here)

                    # (wasteful to recompute this every time)
                    template_metrics["template_log_rewards_mean"] = template_metrics.get("template_log_rewards_mean", 0) + log_rewards.mean().item() / len(template)
                    template_metrics["template_log_rewards_std"] = template_metrics.get("template_log_rewards_std", 0) + log_rewards.std().item() / len(template)

                    fwd_embs.append(torch.flatten(fwd_action_probs).to("cpu").numpy())
                    bck_embs.append(torch.flatten(bck_action_probs).to("cpu").numpy())

                generated_metrics = {}
                trajs, log_rewards = data_source.get_sampled(num=args.num_test_graphs, test=True)

                mean_traj_len, add_edge_idxs, add_node_idxs, stop_idxs = [], [], [], []

                for idx in range(math.ceil(len(trajs) / args.batch_size)):

                    batch_trajs = trajs[idx * args.batch_size : (idx+1) * args.batch_size]

                    traj_lens = torch.tensor([len(t) for t in batch_trajs])
                    flat_batch_trajs = [s for traj in batch_trajs for s in traj]

                    nodes, edges, masks, actions = trajs_to_tensors(flat_batch_trajs)
                    batch_log_rewards = log_rewards[idx * args.batch_size : (idx+1) * args.batch_size]

                    curr_metrics, fwd_action_probs, bck_action_probs = get_metrics(nodes, edges, masks, actions, traj_lens, batch_log_rewards,
                                                                                   base_models, fwd_models, bck_models, args.log_z, log_z_model,
                                                                                   *backward, **config["args"], device=args.device)
                    for k,v in curr_metrics.items():
                        generated_metrics[f"generated_{k}"] = generated_metrics.get(f"generated_{k}", 0) + v / len(template)  # (could save some divides here)

                    # might also be interesting to check the order that edges are added
                    actions = torch.tensor([a for _s, a in trajs[:-1]])
                    sizes = torch.tensor([len(s[0]) for s, _a in trajs[:-1]])

                    mean_traj_len += (traj_lens - 1).tolist()
                    add_edge_idxs.append((actions < sizes**2).sum())
                    add_node_idxs.append((actions == sizes**2).sum())
                    stop_idxs.append((actions == sizes**2 + 1).sum())

                num_nodes, num_edges, num_cliques, num_n_cliques, num_n_cliques_per_node = [], [], [], [], []
                clique_size_dist = [0 for __ in range(args.max_nodes+1)]

                gen_distribution = np.array([[[ 0 for _connectivity in range(2)]
                                                      for _num_nodes_in_one_n_clique in range(7+1)]
                                                          for _num_nodes in range(7+1)])

                graphs = [t[-2][0] for t in trajs]
                generated_metrics["generated_graph_count"] = len(graphs)
                for i, (nodes, edges, masks) in enumerate(graphs):

                    num_nodes.append(torch.sum(torch.sum(nodes, dim=1) > 0, dim=0).item())
                    num_edges.append(torch.sum(edges[:, :, 0], dim=(0, 1)).item())

                    adj_matrix = edges[:num_nodes[-1], :num_nodes[-1], 0]
                    g = nx.from_numpy_array(adj_matrix.cpu().numpy(), edge_attr=None)

                    cliques = nx.algorithms.clique.find_cliques(g)
                    n_cliques = [c for c in cliques if len(c) == args.reward_arg]
                    n_cliques_per_node = np.bincount(sum(n_cliques, []), minlength=num_nodes[-1])

                    num_cliques.append(len(cliques))
                    num_n_cliques.append(len(n_cliques))
                    num_n_cliques_per_node.append(n_cliques_per_node.mean())

                    for size, count in collections.Counter([len(c) for c in cliques]):
                        clique_size_dist[size] += count / len(graphs)

                    gen_distribution[num_nodes[-1]][int(np.sum(n_cliques_per_node == 1))][int(num_nodes[-1]**2 == num_edges[-1])] += 1 / len(graphs)

                num_nodes_dist_counter = collections.Counter(num_nodes)
                generated_metrics["generated_num_nodes"] = [num_nodes_dist_counter[i+1] for i in range(args.max_nodes+1)]
                generated_metrics["generated_clique_size"] = clique_size_dist

                mean_traj_len, add_edge_idxs, add_node_idxs, stop_idxs, num_nodes, num_edges, num_cliques, num_n_cliques, num_n_cliques_per_node = (
                    np.array(mean_traj_len), np.array(add_edge_idxs), np.array(add_node_idxs), np.array(stop_idxs), np.array(num_nodes),
                    np.array(num_edges), np.array(num_cliques), np.array(num_n_cliques), np.array(num_n_cliques_per_node)
                )
                connectivities = (num_nodes ** 2 == num_edges)

                for k, v in zip(["log_rewards", "traj_len", "add_edge_idx", "add_node_idx", "stop_idx",
                                 "num_nodes", "num_edges", "connectivity", "num_cliques", "num_n_cliques", "n_cliques_per_node"],
                                [log_rewards, mean_traj_len, add_edge_idxs, add_node_idxs, stop_idxs,
                                 num_nodes, num_edges, connectivities, num_cliques, num_n_cliques, n_cliques_per_node]):
                    generated_metrics[f"generated_{k}_mean"] = v.mean().item()
                    generated_metrics[f"generated_{k}_std"] = v.std().item()

                eta = 0.001
                # https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
                ks = np.max(np.abs(np.cumsum(gen_distribution) - np.cumsum(tru_distribution)))
                # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence (KL(true || gen))
                kl = np.sum(np.maximum(eta, tru_distribution) * np.log(np.maximum(eta, tru_distribution) / np.maximum(eta, gen_distribution)))
                # https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence (questionable usefulness here?)
                m_distribution = (tru_distribution + gen_distribution) / 2
                js = (np.sum(np.maximum(eta, tru_distribution) * np.log(np.maximum(eta, tru_distribution) / np.maximum(eta, m_distribution))) \
                    + np.sum(np.maximum(eta, gen_distribution) * np.log(np.maximum(eta, gen_distribution) / np.maximum(eta, m_distribution)))) / 2

                generated_metrics["generated_ks"], generated_metrics["generated_kl"], generated_metrics["generated_js"] = ks, kl, js
                generated_metrics["ens_0"], generated_metrics["ens_1"] = data_source.get_log_unnormalised_ens(), data_source.get_log_unnormalised_ens(refl=True)
                generated_metrics["lr"], generated_metrics["random_prob"] = main_optimiser.param_groups[0]['lr'], data_source.random_action_prob

                test_time += time.perf_counter()
                generated_metrics["train_time"], generated_metrics["test_time"] = train_time / (it + 1), test_time / (it + 1)

                metrics = {"iteration": it} | train_metrics | template_metrics | generated_metrics
                train_metrics = {}

                if args.save:

                    with open(f"results/metrics/{it}.pkl", "wb") as f:
                        pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)

                    with open(f"results/batches/{it}.pkl", "wb") as f:
                        pickle.dump((trajs, log_rewards), f, pickle.HIGHEST_PROTOCOL)

                    with open(f"results/batches/{it}_dist.pkl", "wb") as f:
                        pickle.dump(gen_distribution, f, pickle.HIGHEST_PROTOCOL)

                    np.save(f"results/embeddings/{it}_fwd.npy", np.concatenate(fwd_embs, axis=0))
                    np.save(f"results/embeddings/{it}_bck.npy", np.concatenate(bck_embs, axis=0))

                    names = ("stop_model", "node_model", "edge_model")
                    for m, f in zip([base_models[0], *fwd_models, *bck_models, log_z_model],
                                    ["base_model", *("fwd_" + n for n in names), *("bck_" + n for n in names),
                                     *(("n_model", "log_z_model") if args.loss_fn == "tb-max-ent" else ("log_z_model",))]):
                        torch.save(m.state_dict(), f"results/models/{it}_{f}.pt")

                print(f"{metrics['iteration']:<5}: ({metrics['lr']:5.0e}; {metrics['random_prob']:5.2f}) " \
                      f"loss: {metrics['train_combined_loss_mean']:7.2f} (f: {metrics['train_fwd_loss_mean']:7.2f}, b: {metrics['train_bck_loss_mean']:7.2f}) " \
                      f"conn: {metrics['generated_connectivity_mean']:3.1f} r: {metrics['generated_log_rewards_mean']:8.3f} js: {metrics['generated_js']:8.5f} " \
                      f"new: {metrics['new_graphs_above_threshold']:0>2} " \
                      f'''#n: {",".join([f"{i}: {metrics['generated_num_nodes'][i]:0>2}" for i in range(1, 9)])} ''' \
                      f"(n: {metrics['generated_num_nodes_mean']:3.1f} e: {metrics['generated_num_edges_mean']:3.1f}) " \
                      f'''#c: {",".join([f"{i}: {metrics['generated_clique_size'][i]:0>2}" for i in range(1, 9)])}''')

        if (it+1) % 50 == 0:
            data_source.random_action_prob = max(random_prob_min, data_source.random_action_prob * random_prob_decay)

        if args.backward_reset_period > 0 and (it+1) % args.backward_reset_period == 0:
            bck_stop_model = compile_model(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
            bck_node_model = compile_model(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
            bck_edge_model = compile_model(nn.Sequential(nn.Linear(args.num_features*3, args.num_features*3*2), nn.LeakyReLU(), nn.Linear(args.num_features*3*2, 1))).to(args.device)
            bck_models = [bck_stop_model, bck_node_model, bck_edge_model]
            if args.loss_fn == "tb-max-ent":
                n_model = compile_model(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
                bck_models.append(n_model)
            bck_optimiser = torch.optim.Adam(itertools.chain(*(i.parameters() for i in bck_models)), lr=args.learning_rate*10, weight_decay=1e-4)

    if args.meta_test:
        gen_distribution = np.array([[[0 for _connectivity in range(2)] for _num_nodes_in_one_n_clique in range(7+1)] for _num_nodes in range(7+1)])
        trajs, _log_rewards = data_source.get_sampled(num=args.num_test_graphs, test=True)
        for i, (nodes, edges, masks) in enumerate([t[-2][0] for t in trajs]):
            num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0).item()
            num_edges = torch.sum(edges[:, :, 0], dim=(0, 1)).item()
            adj_matrix = edges[:num_nodes, :num_nodes, 0]
            g = nx.from_numpy_array(adj_matrix.cpu().numpy(), edge_attr=None)
            n_cliques = [c for c in nx.algorithms.clique.find_cliques(g) if len(c) == args.reward_arg]
            n_cliques_per_node = np.bincount(sum(n_cliques, []), minlength=num_nodes)
            gen_distribution[num_nodes][int(np.sum(n_cliques_per_node == 1))][int(num_nodes**2 == num_edges)] += 1 / len(trajs)
        eta = 0.001
        kl = np.sum(np.maximum(eta, tru_distribution) * np.log(np.maximum(eta, tru_distribution) / np.maximum(eta, gen_distribution)))
        np.save("results/meta_fitness.npy", kl)

    print("done.")
