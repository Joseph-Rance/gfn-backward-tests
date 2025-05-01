from math import ceil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_transformer_pytorch import GraphTransformer

from data_source import GFNSampler, get_reward_fn_generator, get_smoothed_log_reward

SEED = 1
NUM_BATCHES = 8
BATCH_SIZE = 32
BATCH_ARRGEGATION = 4
OUT_PATH = "results/s/template.npy"
NUM_FEATURES = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

base_model = GraphTransformer(dim=NUM_FEATURES, depth=1, edge_dim=NUM_FEATURES, with_feedforwards=True, gated_residual=True, rel_pos_emb=False)

fwd_stop_model = nn.Sequential(nn.Linear(NUM_FEATURES, NUM_FEATURES*2), nn.LeakyReLU(), nn.Linear(NUM_FEATURES*2, 1))
fwd_node_model = nn.Sequential(nn.Linear(NUM_FEATURES, NUM_FEATURES*2), nn.LeakyReLU(), nn.Linear(NUM_FEATURES*2, 1))
fwd_edge_model = nn.Sequential(nn.Linear(NUM_FEATURES*3, NUM_FEATURES*3*2), nn.LeakyReLU(), nn.Linear(NUM_FEATURES*3*2, 1))
fwd_models = [fwd_stop_model, fwd_node_model, fwd_edge_model]

data_source = GFNSampler(base_model, *fwd_models, get_reward_fn_generator(get_smoothed_log_reward),
                         node_features=NUM_FEATURES, edge_features=NUM_FEATURES,
                         random_action_prob=1, adjust_random=4, max_len=80, max_nodes=8,
                         batch_size=BATCH_SIZE, num_precomputed=0, device="cpu")
data_loader = torch.utils.data.DataLoader(data_source, batch_size=None)

outs = np.array([None]*ceil(NUM_BATCHES / BATCH_ARRGEGATION), dtype=object)
batch = []

for it, (jagged_trajs, _log_rewards) in zip(range(NUM_BATCHES), data_loader):

    for traj in jagged_trajs:

        idx = random.randint(0, len(traj) - 2)
        batch.append(traj[idx])

    if (it+1) % BATCH_ARRGEGATION == 0 or it + 1 == NUM_BATCHES:

        # pad inputs to the same length (this is quite memory intensive)
        nodes = nn.utils.rnn.pad_sequence([n for (n, _e, _m), _a in batch], batch_first=True)
        edges = torch.stack([F.pad(e, (0, 0, 0, nodes.shape[1] - e.shape[1], 0, nodes.shape[1] - e.shape[0]), "constant", 0) for (_n, e, _m), _a in batch])
        masks = torch.stack([F.pad(m, (0, nodes.shape[1] - m.shape[0]), "constant", 0) for (_n, _e, m), _a in batch])
        outs[it // BATCH_ARRGEGATION] = (nodes, edges, masks)

        batch = []

np.save(OUT_PATH, outs, allow_pickle=True)
