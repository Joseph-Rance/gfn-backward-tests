import math
import pickle
import time
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy import spatial
from sklearn import manifold, decomposition
import torch
import torch.nn as nn
from graph_transformer_pytorch import GraphTransformer
import networkx as nx
from rdkit import Chem
from src.drug_design.tasks.util import ReactionTask
from src.graph_building.data_source import GFNSampler, get_reward_fn_generator, get_smoothed_log_reward

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--plot-idx", type=int, default=0)
parser.add_argument("-f", "--filename", type=str, default=None)

args = parser.parse_args()

# for grid task
GRID_SIZE = 7
MULTIPLIER = 16


def get_cliques_log_reward(nodes, edges, n=3, m=10, eta=0.00001, **kwargs):  # reward is ReLU( m * # nodes in exactly 1 n-clique - # edges )
    num_nodes = torch.sum(torch.sum(nodes, dim=2) > 0, dim=1)
    num_edges = torch.sum(edges[:, :, :, 0], dim=(1, 2))
    log_rewards = []
    for i in range(len(nodes)):
        adj_matrix = edges[i, :num_nodes[i], :num_nodes[i], 0]
        g = nx.from_numpy_array(adj_matrix.cpu().numpy(), edge_attr=None)  # does not include create_using=nx.DiGraph, so we convert to an undirected graph
        n_cliques = [c for c in nx.algorithms.clique.find_cliques(g) if len(c) == n]
        n_cliques_per_node = np.bincount(sum(n_cliques, []), minlength=num_nodes[i])
        reward = max(np.sum(n_cliques_per_node == 1) * m - num_edges[i], eta)
        log_rewards.append(math.log(reward))
    return torch.tensor(log_rewards)

# https://gist.github.com/krvajal/1ca6adc7c8ed50f5315fee687d57c3eb
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    window_size = np.abs(int(window_size))
    order = np.abs(int(order))
    if window_size % 2 != 1 or window_size < 1 or window_size < order + 2:
        return 0
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.asmatrix([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")


if args.plot_idx == 0:

    dirs = [i for i in "pqxyz"]

    costs_list = []
    for d in dirs:  # get mean cost last 20
        costs = np.load(f"res_{d}/{d}_cost.npy")
        costs_list.append(costs[:-20].mean())
    print(" ".join([f"{i:.6f}" for i in costs_list]))
    #print(f"{np.array(costs_list).mean():.6f} +/- {np.array(costs_list).std():.6f}")
    # (^^ this std is not between runs)

    scaffolds_list = []
    for d in dirs:  # get mean unique proportion last 20
        scaffolds = np.load(f"res_{d}/{d}_unique_scaffolds.npy")
        per_batch = scaffolds[:, 1] - np.array([0] + scaffolds[:-1, 1].tolist())
        batch_size = scaffolds[1, 0]
        scaffolds_list.append(per_batch[:-20].mean() / batch_size)
    print(" ".join([f"{i:.6f}" for i in scaffolds_list]))
    #print(f"{np.array(scaffolds_list).mean():.6f} +/- {np.array(scaffolds_list).std():.6f}")
    # (^^ this std is not between runs)

elif args.plot_idx == 1:

    plt.rcParams["figure.figsize"] = (6,3)

    ppo = np.array([[0.2650, 0.323840], [0.2519, 0.325178], [0.2553, 0.345454]])
    plt.scatter(ppo[:, 0], ppo[:, 1])
    uniform_line = np.array([[0.3718, 0.482620], [0.3675, 0.423645], [0.3495, 0.406875], [0.2135, 0.157500]])
    plt.scatter(uniform_line[:, 0], uniform_line[:, 1], c="#505050", marker="s")
    me_line = np.array([[0.3815, 0.489063], [0.3350, 0.435938], [0.3257, 0.313167], [0.0625, 0.050000]])
    plt.scatter(me_line[:, 0], me_line[:, 1], c="#505050", marker="D")
    mle_line = np.array([[0.3830, 0.587500], [0.3610, 0.521875], [0.3495, 0.394063], [0.1155, 0.047697]])
    plt.scatter(mle_line[:, 0], mle_line[:, 1], c="#505050", marker="^")

    plt.plot([0.3610, 0.3350, 0.3257, 0.2135, 0.1155], [0.521875, 0.435938, 0.313167, 0.157500, 0.047697], c="#000000", alpha=0.2, zorder=-10, linestyle="--")
    plt.legend(["New", "Uniform", "Maximum Entropy", "Pessimistic"])

elif args.plot_idx == 2:

    plt.rcParams["figure.figsize"] = (5,2)
    SECOND = False
    d = ["a"]

    dist = np.load(f"res_{d}/{d}_lens.npy")
    dist = dist[1:7] / dist.sum()
    if SECOND:
        plt.bar(list(range(len(dist))), [0]*len(dist))
        plt.yticks([])
    plt.bar(list(range(len(dist))), dist)
    plt.ylim((0, 0.9))

elif args.plot_idx == 3:

    plt.rcParams["figure.figsize"] = (6,3)
    comp = ("a", "t")

    with open(f"res_{comp[0]}/{comp[0]}_molecules.pkl", "rb") as f:
        mols_a = pickle.load(f)
    with open(f"res_{comp[1]}/{comp[1]}_molecules.pkl", "rb") as f:
        mols_b = pickle.load(f)

    set_a, set_b = set(mols_a.keys()), set(mols_b.keys())
    shared_mols = set_a.intersection(set_b)
    costs_a = [np.array(mols_a[m]).mean() for m in shared_mols]
    costs_b = [np.array(mols_b[m]).mean() for m in shared_mols]

    rv = plt.violinplot([[np.mean(v) for v in mols_a.values()], costs_a], showextrema=False, vert=False)
    for vr in rv["bodies"]:
        m = np.mean(vr.get_paths()[0].vertices[:, 1])
        r = np.max(vr.get_paths()[0].vertices[:, 1])
        vr.get_paths()[0].vertices[:, 1] = np.clip(vr.get_paths()[0].vertices[:, 1], m, r)
        vr.set_alpha(0.9)

    lv = plt.violinplot([[np.mean(v) for v in mols_b.values()], costs_b], showextrema=False, vert=False)
    for vl in lv["bodies"]:
        m = np.mean(vl.get_paths()[0].vertices[:, 1])
        l = np.min(vl.get_paths()[0].vertices[:, 1])
        vl.get_paths()[0].vertices[:, 1] = np.clip(vl.get_paths()[0].vertices[:, 1], l, m)
        vl.set_alpha(0.9)

    leg = plt.legend(["uniform", "ppo"])
    leg.legend_handles[0].set_facecolor(plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
    leg.legend_handles[0].set_edgecolor(plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
    leg.legend_handles[1].set_facecolor(plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    leg.legend_handles[1].set_edgecolor(plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    plt.yticks([])

elif args.plot_idx == 4:

    plt.rcParams["figure.figsize"] = (6,3)
    comp = "q"

    with open(f"res_{comp}/{comp}_molecules.pkl", "rb") as f:
        mols_a = pickle.load(f)
    set_a = list(set(mols_a.keys()))

    task = ReactionTask("cuda")
    rewards = []
    objs = [Chem.MolFromSmiles(m) for m in set_a]
    for i in trange(math.ceil(len(objs) / 100)):
        obj_props = task.compute_obj_properties(objs[i*100:(i+1)*100])
        rewards += torch.exp(obj_props.squeeze().clamp(min=1e-30).log() * 32).tolist()

    costs_a = np.array([np.array(mols_a[m]).mean() for m in set_a])
    print(costs_a[np.argsort(rewards)[:100]].mean())

elif args.plot_idx == 5:

    reward_dist = np.array([[j*(GRID_SIZE-1-j) + i*(GRID_SIZE-1-i) for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]) ** 5
    reward_dist = np.maximum(reward_dist / np.max(reward_dist), 0.00001)

    fig, ax = plt.subplots()
    ax.matshow(reward_dist, cmap="Grays")
    for (i, j), z in np.ndenumerate(reward_dist):
        ax.text(j, i, "{:0.1f}".format(z), ha="center", va="center", color="white" if (3-i) ** 2 + (3-j) ** 2 < 4 else "black")

    ax.set_xticks([])
    ax.set_yticks([])

elif args.plot_idx == 6:

    reward_dist = np.array([[j*(GRID_SIZE-1-j) + i*(GRID_SIZE-1-i) for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]) ** 5
    reward_dist = np.maximum(reward_dist / np.max(reward_dist), 0.00001)

    total_flow = np.zeros((GRID_SIZE, GRID_SIZE))
    down_flows = np.zeros((GRID_SIZE, GRID_SIZE))
    right_flows = np.zeros((GRID_SIZE, GRID_SIZE))

    for i in range(GRID_SIZE-1, -1, -1):
        for j in range(GRID_SIZE-1, -1, -1):
            total_flow[i, j] = right_flows[i, j] + down_flows[i, j] + reward_dist[i, j]
            if i < GRID_SIZE-1:
                s = None
                down_flows[i, j] = (0. if s[0] == 0 else 1.) + (0. if s[1] == 0 else MULTIPLIER)

elif args.plot_idx == 7:

    reward_dist = np.array([[j*(GRID_SIZE-1-j) + i*(GRID_SIZE-1-i) for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]) ** 5
    reward_dist = np.maximum(reward_dist / np.max(reward_dist), 0.00001)
    reward_dist = reward_dist / np.sum(reward_dist)

    trajs_counts = np.zeros((GRID_SIZE, GRID_SIZE))
    for i in range(GRID_SIZE-1, -1, -1):
        for j in range(GRID_SIZE-1, -1, -1):
            down_flow = trajs_counts[i+1, j] * 1 / (1 + MULTIPLIER * int(j > 0)) if i < GRID_SIZE-1 else 0
            right_flow = trajs_counts[i, j+1] * MULTIPLIER / (MULTIPLIER + int(i > 0)) if j < GRID_SIZE-1 else 0
            trajs_counts[i, j] = reward_dist[i, j] + right_flow + down_flow

    fig, ax = plt.subplots()
    ax.matshow(trajs_counts, cmap="Grays")
    for (i, j), z in np.ndenumerate(trajs_counts):
        ax.text(j, i, "{:0.1f}".format(z), ha="center", va="center", color="white" if trajs_counts[i, j] > trajs_counts.max() / 2 else "black")
    ax.set_xticks([])
    ax.set_yticks([])

elif args.plot_idx == 8:

    N = 2

    reward_dist = np.array([[j*(GRID_SIZE-1-j) + i*(GRID_SIZE-1-i) for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]) ** 5
    reward_dist = np.maximum(reward_dist / np.max(reward_dist), 0.00001)
    reward_dist = reward_dist / np.sum(reward_dist)

    grid = None  # [ paste grid here ]

    stds = np.zeros((GRID_SIZE, GRID_SIZE))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            v = np.array(([grid[i*2+1, j*2]] if i < GRID_SIZE-1 else []) + ([grid[i*2, j*2+1]] if j < GRID_SIZE-1 else []))
            stds[i, j] = (v / v.sum()).std() if len(v) > 1 else 0

    trajs_counts = np.zeros((GRID_SIZE, GRID_SIZE))
    for i in range(GRID_SIZE-1, -1, -1):
        for j in range(GRID_SIZE-1, -1, -1):
            down_flow = trajs_counts[i+1, j] * 1 / (1 + MULTIPLIER * int(j > 0)) if i < GRID_SIZE-1 else 0
            right_flow = trajs_counts[i, j+1] * MULTIPLIER / (MULTIPLIER + int(i > 0)) if j < GRID_SIZE-1 else 0
            trajs_counts[i, j] = reward_dist[i, j] + right_flow + down_flow

    stds_idxs = np.zeros((2*(GRID_SIZE-1)+1,))
    counts_idxs = np.zeros((2*(GRID_SIZE-1)+1,))

    for i in range(GRID_SIZE-1, -1, -1):
        for j in range(GRID_SIZE-1, -1, -1):
            stds_idxs[i+j] += trajs_counts[i, j] * stds[i, j]
            counts_idxs[i+j] += trajs_counts[i, j]
    stds_idxs = stds_idxs / counts_idxs

    x = np.array(counts_idxs.tolist() + [0])
    diffs = x[:-1] - x[1:]
    diffs = diffs / diffs.sum()

    first = stds_idxs[:N].mean()
    last = first2 = 0
    for i, count in enumerate(diffs[:-1]):
        if i < N-1:
            continue
        last += count * stds_idxs[i-N+1:i+1].mean()
        first2 += count * stds_idxs[:N].mean()

    print(first, last)

elif args.plot_idx == 9:

    reward_dist = np.array([[j*(GRID_SIZE-1-j) + i*(GRID_SIZE-1-i) for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]) ** 5
    reward_dist = np.maximum(reward_dist / np.max(reward_dist), 0.00001)

    flows = np.zeros((GRID_SIZE, GRID_SIZE))
    down_probs = np.zeros((GRID_SIZE, GRID_SIZE))
    right_probs = np.zeros((GRID_SIZE, GRID_SIZE))
    stop_probs = np.zeros((GRID_SIZE, GRID_SIZE))

    for i in range(GRID_SIZE-1, -1, -1):
        for j in range(GRID_SIZE-1, -1, -1):
            down_flow = flows[i+1, j] * 1 / (1 + 16 * int(j > 0)) if i < GRID_SIZE-1 else 0
            right_flow = flows[i, j+1] * 16 / (16 + int(i > 0)) if j < GRID_SIZE-1 else 0
            stop_flow = reward_dist[i, j]
            flows[i, j] = down_flow + right_flow + stop_flow
            down_probs[i, j] = down_flow / flows[i, j]
            right_probs[i, j] = right_flow / flows[i, j]
            stop_probs[i, j] = stop_flow / flows[i, j]

    down_log_probs = np.log(np.clip(down_probs, a_min=0.0000000000000000001, a_max=1))
    right_log_probs = np.log(np.clip(right_probs, a_min=0.0000000000000000001, a_max=1))
    stop_log_probs = np.log(np.clip(stop_probs, a_min=0.0000000000000000001, a_max=1))

    flows_unsampled = np.zeros((GRID_SIZE, GRID_SIZE))

    def solve(flows_unsampled, down_log_probs, right_log_probs, stop_log_probs, start=(0, 0), traj_log_prob=0,
              start_flow=flows[0, 0], threshold=math.log(0.0001)):
        total_log_prob = stop_log_probs[start] + traj_log_prob
        if total_log_prob < threshold:
            flows_unsampled[start] += math.exp(total_log_prob) * start_flow
        if start[0] != 6:
            solve(flows_unsampled, down_log_probs, right_log_probs, stop_log_probs, start=(start[0]+1, start[1]),
                  traj_log_prob=traj_log_prob + down_log_probs[start], start_flow=start_flow, threshold=threshold)
        if start[1] != 6:
            solve(flows_unsampled, down_log_probs, right_log_probs, stop_log_probs, start=(start[0], start[1]+1),
                  traj_log_prob=traj_log_prob + right_log_probs[start], start_flow=start_flow, threshold=threshold)

    solve(flows_unsampled, down_log_probs, right_log_probs, stop_log_probs)
    #print(flows_unsampled)
    #flows_unsampled /= flows_unsampled.max()

    fig, ax = plt.subplots()
    ax.matshow(flows_unsampled, cmap="Grays")
    for (i, j), z in np.ndenumerate(flows_unsampled):
        ax.text(j, i, "{:0.3f}".format(z), ha="center", va="center", color="white" if flows_unsampled[i, j] > 0.5 * flows_unsampled.max() else "black")
    ax.set_xticks([])
    ax.set_yticks([])

elif args.plot_idx == 9:

    # count number of undirected graphs with n nodes, with m nodes in
    # exactly 1 fully-connected k-clique (rewrite in c++?)
    k = 3
    m = 10
    eta = 0.00001

    def brute_force(g, counts, rewards, remaining_edges, n, e, max_e, high_reward_graphs):

        if remaining_edges:
            new_edge = remaining_edges.pop(0)
            brute_force(g, counts, rewards, remaining_edges, n, e, max_e, high_reward_graphs)
            g.add_edge(*new_edge)
            brute_force(g, counts, rewards, remaining_edges, n, e+1, max_e, high_reward_graphs)
            g.remove_edge(*new_edge)
            remaining_edges.append(new_edge)

        else:
            k_cliques = [c for c in nx.algorithms.clique.enumerate_all_cliques(g) if len(c) == k]
            num_1_k_clique_nodes = np.sum(np.bincount(sum(k_cliques, []), minlength=n) == 1)
            counts[num_1_k_clique_nodes][0] += 2**n * 3**e
            if max_e == e:
                counts[num_1_k_clique_nodes][1] += 1
                counts[num_1_k_clique_nodes][0] -= 1
            reward = max(float(num_1_k_clique_nodes * m - e), 0)
            reward *= 1e12 / math.exp(0.6 * n ** 2 + 1.2 * n)
            if math.log(max(reward, 0.00000000000001)) > 2.8:  # (for reward index 2)
                high_reward_graphs[n] += 2**n * 3**e
            #reward = 0.5 ** (n ** 2)
            rewards[num_1_k_clique_nodes][0] += reward * 2**n * 3**e
            if max_e == e:
                #reward = 0.8 ** n
                rewards[num_1_k_clique_nodes][1] += reward
                rewards[num_1_k_clique_nodes][0] -= reward

    all_counts, all_rewards = [], []
    high_reward_graphs = [0 for __ in range(8)]

    for n in range(1, 8):

        start = time.perf_counter()

        g = nx.Graph()

        for i in range(n):
            g.add_node(i)

        counts = [[0, 0] for __ in range(8)]
        rewards = [[0, 0] for __ in range(8)]

        edges = [(j, i) for i in range(n) for j in range(i)]

        brute_force(g, counts, rewards, edges, n, 0, n * (n-1) // 2, high_reward_graphs)

        print(f"time: {time.perf_counter() - start:13.6f}; {[sum(i) for i in counts]}; {[sum(i) for i in rewards]}")
        all_counts.append(counts)
        all_rewards.append(rewards)

    print(f"graph counts: {[sum(sum(i) for i in j) for j in all_counts]}")
    print(f"high-reward graph counts: {high_reward_graphs}")

    rewards = np.array(all_rewards)
    print(np.sum(all_rewards))
    np.save("rewards.npy", rewards / np.sum(rewards))

elif args.plot_idx == 10:

    NUM = 7799

    base_model = GraphTransformer(dim=8, depth=2, edge_dim=8, with_feedforwards=True, gated_residual=True, rel_pos_emb=False).to("cuda")
    fwd_stop_model = nn.Sequential(nn.Linear(8, 8*2), nn.LeakyReLU(), nn.Linear(8*2, 1)).to("cuda")
    fwd_node_model = nn.Sequential(nn.Linear(8, 8*2), nn.LeakyReLU(), nn.Linear(8*2, 1)).to("cuda")
    fwd_edge_model = nn.Sequential(nn.Linear(8*3, 8*3*2), nn.LeakyReLU(), nn.Linear(8*3*2, 1)).to("cuda")
    fwd_models = [fwd_stop_model, fwd_node_model, fwd_edge_model]
    fwd_stop_model.load_state_dict(torch.load(f"results/models/fwd_stop_model_{NUM}.pt", weights_only=True))
    base_model.load_state_dict(torch.load(f"results/models/base_model_{NUM}.pt", weights_only=True))
    fwd_node_model.load_state_dict(torch.load(f"results/models/fwd_node_model_{NUM}.pt", weights_only=True))
    fwd_edge_model.load_state_dict(torch.load(f"results/models/fwd_edge_model_{NUM}.pt", weights_only=True))

    data_source = GFNSampler(base_model, *fwd_models, get_reward_fn_generator(get_smoothed_log_reward),
                            node_features=8, edge_features=8,
                            random_action_prob=0, max_len=80, max_nodes=8, base=0.8,
                            batch_size=32, num_precomputed=0, edges_first=False,
                            device="cuda")
    nodes, edges, _mask = data_source.generate_graphs(2)[0]

    num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
    adj_matrix = edges[:num_nodes, :num_nodes, 0]
    g = nx.from_numpy_array(adj_matrix.cpu().numpy(), edge_attr=None)
    print(get_cliques_log_reward(nodes.reshape((-1, *nodes.shape)), edges.reshape((-1, *edges.shape)))[0])

    nx.draw(g)

elif args.plot_idx == 11:

    smooth = lambda x, *args: x

    plt.rcParams["figure.figsize"] = (5,3)
    dirs = [["f0", "f1"], ["g3", "h3"]]
    idxss = [60, 60]
    metrics = ["generated_num_nodes_mean"]

    res = [[] for __ in range(len(metrics))]
    stds = [[] for __ in range(len(metrics))]

    for dir in dirs:

        for r in res:
            r.append([])
        for s in stds:
            s.append([])

        for i in range(99, 10_000, 100):
            mss = []
            for d in dir:
                with open(f"{d}/{i}.pkl", "rb") as input_file:
                    mss.append(pickle.load(input_file))

            for j, metric in enumerate(metrics):
                res[j][-1].append(np.median(np.array([ms[metric] for ms in mss])) if type(mss[0][metric]) in (int, float) else 0)
                stds[j][-1].append((0 if len(mss) < 2 else np.std(np.array([ms[metric] for ms in mss]))) if type(mss[0][metric]) in (int, float) else 0)

    res = np.array(res)
    stds = np.array(stds)

    # mean with window size 10
    N = 10
    idxs = []
    for l in res[0]:
        curr = []
        for i in range(len(l)):
            curr.append(max(0, i+1-N) + np.argsort(l[max(0, i+1-N):1+i])[(1+i-max(0, i+1-N))//2])
        idxs.append(curr)
    std_idxs = []
    for l in stds[0]:
        curr = []
        for i in range(len(l)):
            curr.append(max(0, i+1-N) + np.argsort(l[max(0, i+1-N):1+i])[(1+i-max(0, i+1-N))//2])
        std_idxs.append(curr)

    for i, (lines, std_lines, metric) in enumerate(zip(res, stds, metrics)):
        if i > 0:
            break

        for idx, line, std_idx, std, end in zip(idxs, lines, std_idxs, std_lines, idxss):
            x = [((i+1)*100) for i in range(100)][:end]
            m = smooth(line[idx], 5, 2)[:end]
            s = smooth(std[std_idx], 9, 2)[:end]
            plt.plot(x, m)

        for idx, line, std_idx, std, end in zip(idxs, lines, std_idxs, std_lines, idxss):
            x = [((i+1)*100) for i in range(100)][:end]
            m = smooth(line[idx], 5, 2)[:end]
            s = smooth(std[std_idx], 9, 2)[:end]
            plt.fill_between(x, m+s, m-s, alpha=0.2)

        plt.legend(["uniform", "biased"])

elif args.plot_idx == 12:

    results = np.array([])  # [ paste results here ]

    plt.rcParams["figure.figsize"] = (5,3)

    N = 20

    for line, std in results.T:
        line = np.convolve(line, np.ones(N)/N, mode='valid')
        plt.fill_between([(i+1)*5120 for i in range(len(line))], line+std, line-std, alpha=0.3, label='_nolegend_')

    for line in results.T:
        line = np.convolve(line, np.ones(N)/N, mode='valid')
        #plt.hlines(y=0.5, xmin=0, xmax=5120*(1_000-N+1), color="black", alpha=0.5, zorder=-10, linewidth=0.5)
        plt.plot([(i+1)*5120 for i in range(len(line))], line)

    plt.grid(which='major', color='#AAAAAA', linestyle=':', linewidth=0.5)
    plt.ylim((0.25, 0.525))
    plt.legend(["up/down", "left/right"])

elif args.plot_idx == 13:

    smooth = lambda x, *args: x
    plt.rcParams["figure.figsize"] = (8,5)
    REGEN = False
    SEED = 2

    surface_dirs = ["b0", "b1", "c2", "c3", "d4", "d5", "e6", "e7", "m22", "m23", "m25", "m26"]
    line_dirs = ["tlm0", "tlm4", "tlm5", "resets"]

    if REGEN:

        fwd_embeddings, bck_embeddings = [], []
        losses = []
        colours = [i for i in range(len(surface_dirs) + len(line_dirs)) for __ in range(100)]

        for dir in tqdm(surface_dirs):

            for i in range(99, 10_000, 100):
                fwd_embeddings.append(np.load(f"{dir}/{i}_fwd.npy"))
                bck_embeddings.append(np.load(f"{dir}/{i}_bck.npy"))

            for i in range(99, 10_000, 100):
                with open(f"{dir}/metrics (9)/{i}.pkl", "rb") as input_file:
                    losses.append(pickle.load(input_file)["template_fwd_loss_mean"])

        idx = len(losses)

        for dir in tqdm(line_dirs):

            for i in range(99, 10_000, 100):
                fwd_embeddings.append(np.load(f"{dir}/{i}_fwd.npy"))
                bck_embeddings.append(np.load(f"{dir}/{i}_bck.npy"))

            for i in range(99, 10_000, 100):
                with open(f"{dir}/metrics (9)/{i}.pkl", "rb") as input_file:
                    losses.append(pickle.load(input_file)["template_fwd_loss_mean"])


        fwd_embeddings = np.stack(fwd_embeddings)
        bck_embeddings = np.stack(bck_embeddings)
        losses = np.array(losses)
        colours = np.array(colours)

        print(fwd_embeddings.shape, bck_embeddings.shape, losses.shape, colours.shape)

        print("PCA")

        pca = decomposition.PCA(n_components=50)
        pca_fwd_embeddings = pca.fit_transform(fwd_embeddings)
        pca_bck_embeddings = pca.fit_transform(bck_embeddings)

        print("tSNE")

        tsne = manifold.TSNE(n_components=1, random_state=SEED)
        tsne_fwd_embeddings = tsne.fit_transform(pca_fwd_embeddings).flatten()
        tsne_bck_embeddings = tsne.fit_transform(pca_bck_embeddings).flatten()

        losses = np.log(losses)
        tsne_fwd_embeddings_surface = tsne_fwd_embeddings[:idx]
        tsne_bck_embeddings_surface = tsne_bck_embeddings[:idx]
        losses_surface = losses[:idx]
        colours_surface = colours[:idx]

        # clip really high losses - mainly from iteration 0
        tsne_fwd_embeddings_surface = tsne_fwd_embeddings_surface[losses_surface < 4.5]
        tsne_bck_embeddings_surface = tsne_bck_embeddings_surface[losses_surface < 4.5]
        colours_surface = colours_surface[losses_surface < 4.5]
        losses_surface = losses_surface[losses_surface < 4.5]

        np.save("fwd_tsne_frozen.npy", tsne_fwd_embeddings_surface)
        np.save("bck_tsne_frozen.npy", tsne_bck_embeddings_surface)
        np.save("losses_frozen.npy", losses_surface)
        np.save("colours_frozen.npy", colours_surface)

        tsne_fwd_embeddings_line = tsne_fwd_embeddings[idx:]
        tsne_bck_embeddings_line = tsne_bck_embeddings[idx:]
        losses_line = losses[idx:]
        colours_line = colours[idx:]

        tsne_fwd_embeddings_line = tsne_fwd_embeddings_line[losses_line < 4.5]
        tsne_bck_embeddings_line = tsne_bck_embeddings_line[losses_line < 4.5]
        colours_line = colours_line[losses_line < 4.5]
        losses_line = losses_line[losses_line < 4.5]

        np.save("fwd_tsne_tlm.npy", tsne_fwd_embeddings_line)
        np.save("bck_tsne_tlm.npy", tsne_bck_embeddings_line)
        np.save("losses_tlm.npy", losses_line)
        np.save("colours_tlm.npy", colours_line)

    else:

        tsne_fwd_embeddings = np.load("fwd_tsne_frozen.npy")
        tsne_bck_embeddings = np.load("bck_tsne_frozen.npy")
        losses = np.load("losses.npy")
        colours = np.load("colours.npy")

        tsne_fwd_embeddings_line = np.load("fwd_tsne_tlm.npy")
        tsne_bck_embeddings_line = np.load("bck_tsne_tlm.npy")
        losses_line = np.load("losses_tlm.npy")
        colours_line = np.load("colours_tlm.npy")

    for j in range(len(surface_dirs)):

        tmp_losses = losses[colours == j]
        tmp_tsne_fwd_embeddings = tsne_fwd_embeddings[colours == j]

        order = np.argsort(tmp_tsne_fwd_embeddings)
        tmp_tsne_fwd_embeddings = tmp_tsne_fwd_embeddings[order]
        tmp_losses = tmp_tsne_fwd_embeddings[order]

        N = 13
        curr = []
        for i in range(len(tmp_losses)):
            curr.append(max(0, i-N//2) + np.argsort(tmp_losses[max(0, i-N//2):i+math.ceil(N / 2)])[(i+math.ceil(N / 2)-max(0, i-N//2))//2])

        losses[colours == j] = smooth(tmp_losses[curr], 5, 1)
        tsne_fwd_embeddings[colours == j] = smooth(tmp_tsne_fwd_embeddings, 7, 1)

    for j in range(len(surface_dirs), len(surface_dirs)+len(line_dirs)):

        tmp_losses = losses_line[colours_line == j]
        tmp_tsne_fwd_embeddings = tsne_fwd_embeddings_line[colours_line == j]
        tmp_tsne_bck_embeddings = tsne_bck_embeddings_line[colours_line == j]

        order = np.argsort(tmp_tsne_bck_embeddings)
        tmp_tsne_fwd_embeddings = tmp_tsne_fwd_embeddings[order]
        tmp_tsne_bck_embeddings = tmp_tsne_bck_embeddings[order]
        tmp_losses = tmp_tsne_fwd_embeddings[order]

        N = 13
        curr = []
        for i in range(len(tmp_losses)):
            curr.append(max(0, i-N//2) + np.argsort(tmp_losses[max(0, i-N//2):i+math.ceil(N / 2)])[(i+math.ceil(N / 2)-max(0, i-N//2))//2])

        losses_line[colours_line == j] = smooth(tmp_losses[curr], 5, 1)
        tsne_fwd_embeddings_line[colours_line == j] = smooth(tmp_tsne_fwd_embeddings, 7, 1)
        tsne_bck_embeddings_line[colours_line == j] = smooth(tmp_tsne_bck_embeddings, 7, 1)

    NUM = 1_000

    #tsne_fwd_embeddings = np.concatenate((tsne_fwd_embeddings, tsne_fwd_embeddings_line))
    #tsne_bck_embeddings = np.concatenate((tsne_bck_embeddings, tsne_bck_embeddings_line))
    #losses = np.concatenate((losses, losses_line))

    x = np.arange(tsne_fwd_embeddings.min(), tsne_fwd_embeddings.max(), (tsne_fwd_embeddings.max() - tsne_fwd_embeddings.min()) / NUM)
    y = np.arange(tsne_bck_embeddings.min(), tsne_bck_embeddings.max(), (tsne_bck_embeddings.max() - tsne_bck_embeddings.min()) / NUM)
    X, Y = np.meshgrid(x, y)

    points = np.stack((tsne_fwd_embeddings, tsne_bck_embeddings)).T
    tree = spatial.KDTree(points)

    k = 20
    Z = np.zeros(X.shape)
    for i in trange(X.shape[0]):
        for j in range(X.shape[1]):
            q = tree.query((X[i, j], Y[i, j]), k=k)
            Z[i, j] = np.sum(losses[q[1]] * q[0] / q[0].sum())

    #triangles = mtri.Triangulation(X.flatten(), Y.flatten())
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1, projection='3d')
    #fig, ax = plt.subplots()
    #ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=Z.flatten())
    #ax.plot_trisurf(triangles, Z.flatten(), cmap='viridis')
    #ax.view_init(elev=65, azim=315, roll=0)
    #ax.view_init(elev=90, azim=315, roll=0)

    #CS = ax.contour(X, Y, Z)#, levels=[3.9, 3.95, 4.0, 4.1])
    #ax.clabel(CS, fontsize=8)

    fig, ax = plt.subplots()

    c = ax.pcolormesh(X, Y, Z, cmap='viridis', vmin=Z.min(), vmax=Z.max())
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])

    mins = x[Z.argmin(axis=1)]

    plt.scatter([-62.122185, -62.00261, -76.03407], [-47.131955, -47.5321, -52.09945],
                zorder=10,color="white", alpha=1, edgecolors='none', s=2 * plt.rcParams['lines.markersize'] ** 2)
    plt.scatter([26.36625, 9.09114, -62.121902], [-26.34578, -39.469627, -47.332283],
                zorder=10, color="black", alpha=1, edgecolors='none', s=2 * plt.rcParams['lines.markersize'] ** 2)

    plt.xticks([], [])
    plt.yticks([], [])

if args.filename:
    plt.savefig(args.filename, dpi=300, bbox_inches="tight")
