import collections
import itertools
import argparse
import random
import time
import sys
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from graph_transformer_pytorch import GraphTransformer

from data_source import (
    GFNSampler,
    get_reward_fn_generator,
    get_smoothed_log_overfit_reward,
    get_uniform_counting_log_reward,
    get_cliques_log_reward
)
from gfn import get_loss_to_uniform_backward, get_metrics

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
    loss_aligned
)

with open("run_command.sh", "w") as f:
    f.write(" ".join(sys.argv))

parser = argparse.ArgumentParser()

# general
parser.add_argument("-s", "--seed", type=int, default=1)
parser.add_argument("-d", "--device", type=str, default="cuda", help="generally 'cuda' or 'cpu'")
parser.add_argument("-e", "--save", action="store_true", default=False, help="whether to save outputs to a file")
parser.add_argument("-c", "--cycle-len", type=int, default=100, help="how often to log/checkpoint (number of batches)")
parser.add_argument("-t", "--num-test-graphs", type=int, default=64, help="number of graphs to generate for estimating metrics")

# env
parser.add_argument("-r", "--reward-idx", type=int, default=2, help="index of reward function to use")
parser.add_argument("-b", "--reward-arg", type=float, default=0.8, help="base for exponent used in reward calculation; size of cliques to search for")  # SET TO 3 FOR CLIQUES!

# model
parser.add_argument("-f", "--num-features", type=int, default=16, help="number of features used to represent each node/edge (min 2)")
parser.add_argument("-y", "--depth", type=int, default=2, help="depth of the transformer model")
parser.add_argument("-g", "--max-nodes", type=int, default=9, help="maximum number of nodes in a generated graph")
parser.add_argument("-q", "--random-action-template", type=int, default=2, help="index of the random action config to use (see code)")
parser.add_argument("-z", "--log-z", type=float, default=None, help="constant value of log(z) to use (learnt if None)")
parser.add_argument("-i", "--backward_init", type=str, default="random", help="how to initialise the backward policy; one of {random, uniform, <directory name>}")
parser.add_argument("-j", "--history-bounds", type=int, default=1, help="controls how much the MDP is like a tree")

# training
parser.add_argument("-l", "--loss-fn", type=str, default="tb-uniform", help="loss function for training (e.g. TB + uniform backward policy)")
parser.add_argument("-v", "--loss-arg-a", type=float, default=1)
parser.add_argument("-u", "--loss-arg-b", type=float, default=1)
parser.add_argument("-w", "--loss-arg-c", type=float, default=1)
parser.add_argument("-m", "--batch-size", type=int, default=128)
parser.add_argument("-p", "--num-precomputed", type=int, default=0, help="number of trajectories from precomputed, high-reward graphs")
parser.add_argument("-a", "--learning-rate", type=float, default=0.0005)
parser.add_argument("-n", "--max-update-norm", type=float, default=99.9)
parser.add_argument("-k", "--num-batches", type=int, default=10_000)
parser.add_argument("-x", "--backward-reset-period", type=int, default=-1, help="how often to reset the backward policy (-1 for no resets)")
parser.add_argument("-o", "--print-metrics", type=str, default="", help="keys of metrics to print")

args = parser.parse_args()

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
    ]# TODO: test this converges
    "tb-loss-aligned": (loss_aligned, {"parameterise_backward": False, "args": {"iters": args.loss_arg_a, "std_mult": args.loss_arg_b}})  # aligned to handmade backward policy
}

backward, config = configs[args.loss_fn]
get_loss = lambda *pargs, **kwargs: get_loss_to_uniform_backward(*backward)(*pargs, **config["args"], **kwargs)
parameterise_backward = config["parameterise_backward"]

reward_fns = [get_uniform_counting_log_reward, get_smoothed_log_overfit_reward, get_cliques_log_reward]
reward_fn = reward_fns[args.reward_idx]

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
                  (0.8, 0.99, 0.15)]
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

    mean_metrics = {}
    
    train_time = test_time = 0

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

        for k,v in curr_metrics.items():
            mean_metrics[f"train_{k}"] = mean_metrics.get(f"train_{k}", 0) + v / args.cycle_len

        mean_metrics["train_norm"] = mean_metrics.get("train_norm", 0) + norm / args.cycle_len

        with torch.no_grad():

            train_time += time.perf_counter()
            test_time -= time.perf_counter()

            if args.save and (it+1)%args.cycle_len == 0:

                for m in (*base_models, log_z_model, *fwd_models, *bck_models):
                    m.eval()

                template_metrics = {}

                fwd_embs, bck_embs = []
                template = np.load("results/s/template.npy", allow_pickle=True)
                for nodes, edges, masks, actions, traj_lens, log_rewards in template:  # TODO: correctly generate these inc importance sampling trajs

                    curr_metrics, fwd_action_probs, bck_action_probs = get_metrics(nodes, edges, masks, actions, traj_lens, log_rewards,
                                                                                   base_models, fwd_models, bck_models, args.log_z, log_z_model,
                                                                                   *backward, **config["args"], device=args.device)
                    for k,v in curr_metrics.items():
                        template_metrics[f"template_{k}"] = template_metrics.get(f"template_{k}", 0) + v / len(template)  # (could save some divides here)

                    # (wasteful to recompute this every time)
                    template_metrics["template_log_rewards_mean"] = template_metrics.get("template_log_rewards_mean", 0) + log_rewards.mean().item() / len(template)
                    template_metrics["template_log_rewards_mean"] = template_metrics.get("template_log_rewards_mean", 0) + log_rewards.std().item() / len(template)

                    fwd_embs.append(torch.flatten(fwd_action_probs).to("cpu").numpy())
                    bck_embs.append(torch.flatten(bck_action_probs).to("cpu").numpy())

                generated_metrics = {}
                trajs, log_rewards = data_source.get_sampled(num=args.num_test_graphs, test=True)

                traj_lens, add_edge_idxs, add_node_idxs, stop_idxs = [], [], [], []

                for t in trajs:

                    # TODO t to args below

                    curr_metrics, fwd_action_probs, bck_action_probs = get_metrics(nodes, edges, masks, actions, traj_lens, log_rewards,
                                                                                   base_models, fwd_models, bck_models, args.log_z, log_z_model,
                                                                                   *backward, **config["args"], device=args.device)
                    for k,v in curr_metrics.items():
                        template_metrics[f"generated_{k}"] = template_metrics.get(f"generated_{k}", 0) + v / len(template)  # (could save some divides here)

                    # might also be interesting to check the order that edges are added
                    actions = torch.tensor([a for _s, a in trajs[:-1]])
                    sizes = torch.tensor([len(s[0]) for s, _a in trajs[:-1]])

                    traj_lens.append(len(t) - 1)
                    add_edge_idxs.append((actions < sizes**2).sum())
                    add_node_idxs.append((actions == sizes**2).sum())
                    stop_idxs.append((actions == sizes**2 + 1).sum())

                num_nodes, num_edges, num_cliques, num_n_cliques, num_n_cliques_per_node = [], [], [], [], []
                clique_size_dist = [0 for __ in range(args.max_nodes+1)]

                graphs = [t[-2][0] for t in trajs]
                template_metrics["generated_graph_count"] = len(graphs)
                for i, (nodes, edges, masks) in enumerate(graphs):

                    num_nodes.append(torch.sum(torch.sum(nodes, dim=1) > 0, dim=0))
                    num_edges.append(torch.sum(edges[:, :, 0], dim=(0, 1)))

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

                # <BEGIN UNFINISHED>

                    # TODO: tanimoto similarity and mean reward for top k dissimilar graphs
                    # TODO: num nodes, connectivities, m nodes in exactly 1 fully-connected k-clique for distribution similarities (ks, klm, js)

                num_nodes_dist_counter = collections.Counter(num_nodes)
                template_metrics["generated_num_nodes"] = [num_nodes_dist_counter[i+1] for i in range(args.max_nodes+1)]
                template_metrics["generated_clique_size"] = clique_size_dist

                traj_lens, add_edge_idxs, add_node_idxs, stop_idxs, num_nodes, num_edges, num_cliques, num_n_cliques, num_n_cliques_per_node = (
                    np.array(traj_lens), np.array(add_edge_idxs), np.array(add_node_idxs), np.array(stop_idxs), np.array(num_nodes),
                    np.array(num_edges), np.array(num_cliques), np.array(num_n_cliques), np.array(num_n_cliques_per_node)
                )
                connectivities = (num_nodes ** 2 == num_edges)

                for k, v in zip(["log_rewards", "traj_lens", "add_edge_idx", "add_node_idx", "stop_idx",
                                 "num_nodes", "num_edges", "connectivity", "num_cliques", "num_n_cliques", "n_cliques_per_node"],
                                [log_rewards, traj_lens, add_edge_idxs, add_node_idxs, stop_idxs,
                                 num_nodes, num_edges, connectivities, num_cliques, num_n_cliques, n_cliques_per_node]):
                    template_metrics[f"generated_{k}_mean"] = v.mean().item()
                    template_metrics[f"generated_{k}_std"] = v.std().item()







                # assume that samples are uniformly generated in these buckets (questionalble)
                # does this make it a lower bound?
                gen_distribution = np.array([0 for __ in range(1, args.max_nodes+1) for c in ["d", "c"]], dtype=float)
                for n, c in zip(test_node_counts, test_connectivities):
                    gen_distribution[2*(n-1) + c] += 1
                gen_distribution /= len(test_connectivities)

                if args.reward_idx == 2:
                    tru_distribution = np.array([v for n in range(1, args.max_nodes+1) for v in [(1 - 2 ** (- n ** 2)) / 8, (2 ** (- n ** 2)) / 8]])
                else:
                    s = args.reward_arg*(1-args.reward_arg**8)/(1-args.reward_arg)
                    tru_distribution = np.array([v for n in range(1, args.max_nodes+1) for v in [0, (args.reward_arg ** n)/s]])

                #time:      0.000731; [1, 0, 0, 0, 0, 0, 0, 0, 0]; [1e-05, 0, 0, 0, 0, 0, 0, 0, 0]
                #time:      0.000433; [2, 0, 0, 0, 0, 0, 0, 0, 0]; [2e-05, 0, 0, 0, 0, 0, 0, 0, 0]
                #time:      0.000172; [7, 0, 0, 1, 0, 0, 0, 0, 0]; [7.000000000000001e-05, 0, 0, 27.0, 0, 0, 0, 0, 0]
                #time:      0.000859; [42, 0, 6, 16, 0, 0, 0, 0, 0]; [0.00042000000000000045, 0, 90.0, 420.0, 0, 0, 0, 0, 0]
                #time:      0.014926; [439, 30, 240, 300, 15, 0, 0, 0, 0]; [0.004389999999999989, 60.0, 3300.0, 7510.0, 510.0, 0, 0, 0, 0]
                #time:      0.593345; [8933, 3120, 10680, 8600, 1095, 0, 340, 0, 0]; [0.08932999999999806, 1260.0203999999535, 128370.0, 199980.0, 35655.0, 0, 17730.0, 0, 0]
                #time:     47.605963; [439531, 375165, 688800, 442785, 92505, 24591, 33775, 0, 0]; [4.395310000002771, 8193.676049947462, 6726930.0, 9196215.0, 2824605.0, 975849.0, 1714125.0, 0, 0]
                # ADJUST FOR REPEATED EDGES WITH DIGRAPH IN DIST??

                eta = 0.001
                # https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
                ks = np.max(np.abs(np.cumsum(gen_distribution) - np.cumsum(tru_distribution)))
                # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence (KL(true || gen))
                kl = np.sum(np.maximum(eta, tru_distribution) * np.log(np.maximum(eta, tru_distribution) / np.maximum(eta, gen_distribution)))
                # https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence (questionable usefulness here?)
                m_distribution = (tru_distribution + gen_distribution) / 2
                js = (np.sum(np.maximum(eta, tru_distribution) * np.log(np.maximum(eta, tru_distribution) / np.maximum(eta, m_distribution))) \
                    + np.sum(np.maximum(eta, gen_distribution) * np.log(np.maximum(eta, gen_distribution) / np.maximum(eta, m_distribution)))) / 2













                data_source.get_log_unnormalised_ens()
                data_source.get_log_unnormalised_ens(refl=True)
                train_time / (it+1)
                test_time / (it+1)
                main_optimiser.param_groups[0]['lr']
                data_source.random_action_prob












                if args.save:

                    # save metrics

                    np.save(f"results/batches/nodes_{it}_{i}.npy", nodes.to("cpu").numpy())
                    np.save(f"results/batches/edges_{it}_{i}.npy", edges.to("cpu").numpy())
                    np.save(f"results/batches/masks_{it}_{i}.npy", edges.to("cpu").numpy())

                    fwd_embs, bck_embs

                    names = ("stop_model", "node_model", "edge_model")
                    for m, f in zip([base_models[0], *fwd_models, *bck_models, log_z_model],
                                    ["base_model", *("fwd_" + n for n in names), *("bck_" + n for n in names), "log_z_model"]):
                        torch.save(m.state_dict(), f"results/models/{f}_{it}.pt")

                args.print_metrics  # TODO: also set default

                print(
                    f"{it: <5} loss: {sum_loss.item():7.2f}" \
                      + (f" (fwd: {sum_loss_fwd.item():7.2f}, bck: {sum_loss_bck.item():7.2f})" if parameterise_backward else "") + \
                    f"; norm: {norm:6.3f}; lr: {main_optimiser.param_groups[0]['lr']:9.7f}; " \
                    f"log(z): {metrics['log_z']:6.3f}; " \
                    f"mean log reward: {test_mean_log_reward:8.3f} ({mean_log_reward:8.3f}); " \
                    f"randomness: {data_source.random_action_prob:5.3f}; "
                        f"connected: {test_mean_connected_prop:4.2f} ({mean_connected_prop:4.2f}); " \
                    f"({mean_num_nodes:3.1f}; {len(graphs)}), {', '.join([f'{i}: {test_node_count_distribution[i]:0>2}' for i in range(1, 9)])}; " \
                    f"ks: {ks:8.5f}; kl: {kl:8.5f}; js: {js:8.5f}"
                )



                # <END UNFINISHED>
            
            test_time += time.perf_counter()

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


    print("done.")
