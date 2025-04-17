import collections
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
from graph_transformer_pytorch import GraphTransformer

from data_source import GFNSampler, get_reward_fn_generator, get_smoothed_log_reward
from gfn import get_tb_loss_manual


SEED = 43
DEVICE = "cuda"
SAVE = False
CYCLE = 5
NUM_TEST_GRAPHS = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# env
BASE = 0.8

# model
NUM_NODE_FEATURES = NUM_EDGE_FEATURES = 10
DEPTH = 1
MAX_LEN = 80
MAX_NODES = 8
RANDOM_PROB, RANDOM_PROB_DECAY, MIN_RANDOM_PROB = 1, 0.99, 0.1

cat_edge_features = NUM_EDGE_FEATURES + 2*NUM_NODE_FEATURES
base_model = torch.compile(GraphTransformer(dim=NUM_NODE_FEATURES, depth=DEPTH, edge_dim=NUM_EDGE_FEATURES, with_feedforwards=True, gated_residual=True, rel_pos_emb=False)).to(DEVICE)
stop_model = torch.compile(nn.Sequential(nn.Linear(NUM_NODE_FEATURES, NUM_NODE_FEATURES*2), nn.LeakyReLU(), nn.Linear(NUM_NODE_FEATURES*2, 1))).to(DEVICE)
node_model = torch.compile(nn.Sequential(nn.Linear(NUM_NODE_FEATURES, NUM_NODE_FEATURES*2), nn.LeakyReLU(), nn.Linear(NUM_NODE_FEATURES*2, 1))).to(DEVICE)
edge_model = torch.compile(nn.Sequential(nn.Linear(cat_edge_features, cat_edge_features*2), nn.LeakyReLU(), nn.Linear(cat_edge_features*2, 1))).to(DEVICE)
log_z_model = torch.compile(nn.Linear(1, 1, bias=False)).to(DEVICE)

# training
BATCH_SIZE = 32
NUM_PRECOMPUTED = 16
LR = 0.00001, 0.0001
REG = 0, 0
MAX_NORM = 100
NUM_ROUNDS = 50_000

main_optimiser = torch.optim.Adam(base_model.parameters(), lr=LR[0], weight_decay=REG[0])
fwd_optimiser = torch.optim.Adam(itertools.chain(stop_model.parameters(), node_model.parameters(), edge_model.parameters()), lr=LR[1], weight_decay=REG[1])
log_z_optimiser = torch.optim.Adam(log_z_model.parameters(), lr=LR[1], weight_decay=REG[1])

main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(main_optimiser, T_max=NUM_ROUNDS)
fwd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fwd_optimiser, T_max=NUM_ROUNDS)
log_z_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(log_z_optimiser, T_max=NUM_ROUNDS)

if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    reward_fn_generator = get_reward_fn_generator(get_smoothed_log_reward, base=BASE)

    data_source = GFNSampler(base_model, stop_model, node_model, edge_model, reward_fn_generator,
                             node_features=NUM_NODE_FEATURES, edge_features=NUM_EDGE_FEATURES,
                             random_action_prob=RANDOM_PROB, max_len=MAX_LEN, max_nodes=MAX_NODES, base=BASE,
                             batch_size=BATCH_SIZE, num_precomputed=NUM_PRECOMPUTED,
                             device=DEVICE)
    data_loader = torch.utils.data.DataLoader(data_source, batch_size=None)

    sum_loss = mean_log_reward = mean_connected_prop = mean_num_nodes = 0

    for it, (jagged_trajs, log_rewards) in zip(range(NUM_ROUNDS), data_loader):

        loss, metrics = get_tb_loss_manual(base_model, stop_model, node_model, edge_model, log_z_model, jagged_trajs, log_rewards, device=DEVICE)
        loss.backward()

        sq_norm = 0
        for m in (base_model, stop_model, node_model, edge_model, log_z_model):
            sq_norm += min(MAX_NORM, torch.nn.utils.clip_grad_norm_(m.parameters(), MAX_NORM).item()) ** 2

        main_optimiser.step()
        main_optimiser.zero_grad()
        fwd_optimiser.step()
        fwd_optimiser.zero_grad()
        log_z_optimiser.step()
        log_z_optimiser.zero_grad()

        main_scheduler.step()
        fwd_scheduler.step()
        log_z_scheduler.step()

        sum_loss += loss.detach() / CYCLE
        mean_log_reward += metrics["mean_log_reward"] / CYCLE
        mean_connected_prop += metrics["connected_prop"] / CYCLE
        mean_num_nodes += metrics["mean_num_nodes"] / CYCLE

        with torch.no_grad():

            if (it+1)%CYCLE == 0:

                test_mean_log_reward = test_mean_connected_prop = 0
                test_node_counts = []

                graphs = data_source.generate_graphs(NUM_TEST_GRAPHS)
                for i, (nodes, edges, masks) in enumerate(graphs):
                    num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
                    num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))

                    test_mean_log_reward += get_smoothed_log_reward(nodes.reshape((1, *nodes.shape)), edges.reshape((1, *edges.shape)), alpha=1_000_000).item()
                    test_mean_connected_prop += float(num_edges == num_nodes**2)
                    test_node_counts.append(num_nodes.item())

                    if SAVE:
                        np.save(f"results/batches/nodes_{it}_{i}.npy", nodes.to("cpu").numpy())
                        np.save(f"results/batches/edges_{it}_{i}.npy", edges.to("cpu").numpy())
                        np.save(f"results/batches/masks_{it}_{i}.npy", edges.to("cpu").numpy())

                test_mean_log_reward /= NUM_TEST_GRAPHS
                test_mean_connected_prop /= NUM_TEST_GRAPHS

                test_node_count_distribution = collections.Counter(test_node_counts)

                ens_0 = data_source.get_log_unnormalised_ens()
                ens_1 = data_source.get_log_unnormalised_ens(refl=True)

                # 0.8 should have 1: 13, 2: 10, 3: 08, 4: 07, 5: 05, 6: 04, 7: 03, 8: 03
                print(
                    f"{it: <5} loss: {sum_loss.item():8.2f}; " \
                    f"norm: {(sq_norm**0.5):6.3f}; " \
                    f"log(z): {metrics['log_z']:6.3f}; " \
                    f"mean log reward: {test_mean_log_reward:8.3f} ({mean_log_reward:8.3f}); " \
                    f"connected: {test_mean_connected_prop:4.2f} ({mean_connected_prop:4.2f}); " \
                    f"ens_0: [{', '.join([f'{i.item():6.2f}' for i in ens_0])}]; " \
                    f"ens_1: [{', '.join([f'{i.item():6.2f}' for i in ens_1[1:]])}]; " \
                    f"({mean_num_nodes:3.1f}; {len(graphs)}), {', '.join([f'{i}: {test_node_count_distribution[i]:0>2}' for i in range(1, 9)])}"
                )

                sum_loss = mean_log_z = mean_log_reward = mean_connected_prop = mean_num_nodes = 0

                data_source.random_action_prob = max(MIN_RANDOM_PROB, data_source.random_action_prob * RANDOM_PROB_DECAY)

                if SAVE:
                    torch.save(base_model.state_dict(), f"results/models/base_model_{it}.pt")
                    torch.save(stop_model.state_dict(), f"results/models/stop_model_{it}.pt")
                    torch.save(node_model.state_dict(), f"results/models/node_model_{it}.pt")
                    torch.save(edge_model.state_dict(), f"results/models/edge_model_{it}.pt")
                    torch.save(log_z_model.state_dict(), f"results/models/log_z_model_{it}.pt")

    print("done.")
