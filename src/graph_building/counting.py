"""count number of undirected graphs with n nodes, with m nodes in exactly 1 fully-connected k-clique"""

import time
import numpy as np
import networkx as nx


k = 3
m = 10
eta = 0.00001

def brute_force(g, counts, rewards, remaining_edges, n, e):

    if remaining_edges:
        new_edge = remaining_edges.pop(0)
        brute_force(g, counts, rewards, remaining_edges, n, e)
        g.add_edge(*new_edge)
        brute_force(g, counts, rewards, remaining_edges, n, e+1)
        g.remove_edge(*new_edge)
        remaining_edges.append(new_edge)
    else:
        k_cliques = [c for c in nx.algorithms.clique.enumerate_all_cliques(g) if len(c) == k]
        num_1_k_clique_nodes = np.sum(np.bincount(sum(k_cliques, []), minlength=n) == 1)
        counts[num_1_k_clique_nodes] += 1
        rewards[num_1_k_clique_nodes] += max(float(num_1_k_clique_nodes * m - e), eta)

all_counts, all_rewards = [], []

for n in range(1, 8):

    start = time.perf_counter()

    g = nx.Graph()

    for i in range(n):
        g.add_node(i)

    counts = [0 for __ in range(9)]
    rewards = [0 for __ in range(9)]

    edges = [(j, i) for i in range(n) for j in range(i)]

    brute_force(g, counts, rewards, edges, n, 0)

    print(f"time: {time.perf_counter() - start:13.6f}; {counts}; {rewards}")
    all_counts.append(counts)
    all_rewards.append(rewards)

print(f"\n\n{all_counts}")
print(f"\n\n{all_rewards}")
np.save("counts.npy", np.array(all_counts))
np.save("rewards.npy", np.array(all_rewards))
