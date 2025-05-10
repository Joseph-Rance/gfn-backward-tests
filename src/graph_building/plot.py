import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
from graph_transformer_pytorch import GraphTransformer

from data_source import GFNSampler, get_reward_fn_generator, get_smoothed_log_reward


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

graphs = data_source.generate_graphs(32)

for nodes, edges, _mask in graphs:

    num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
    adj_matrix = edges[:num_nodes, :num_nodes, 0]
    g = nx.from_numpy_array(adj_matrix.cpu().numpy(), edge_attr=None)

    print(get_cliques_log_reward(nodes.reshape((-1, *nodes.shape)), edges.reshape((-1, *edges.shape)))[0])

    nx.draw(g)
    plt.show()

plt.show()

print("done.")