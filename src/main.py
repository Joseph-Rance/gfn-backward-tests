import collections
import itertools
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from graph_transformer_pytorch import GraphTransformer

from data_source import GFNSampler, get_reward_fn_generator, get_smoothed_log_reward
from gfn import get_tb_loss_uniform, get_tb_loss_add_node_mult, get_tb_loss_tlm


parser = argparse.ArgumentParser()

# general
parser.add_argument("-s", "--seed", default=1)
parser.add_argument("-v", "--device", default="cuda", help="generally 'cuda' or 'cpu'")
parser.add_argument("-o", "--save", default=False, help="whether to save outputs to a file")
parser.add_argument("-c", "--cycle-len", default=5, help="how often to log/checkpoint (number of batches)")
parser.add_argument("-t", "--num-test-graphs", default=0, help="number of graphs to generate for estimating metrics")

# env
parser.add_argument("-b", "--base", default=0.8, help="base for exponent used in reward calculation")

# model
parser.add_argument("-f", "--num-features", default=10, help="number of features used to represent each node/edge (min 2)")
parser.add_argument("-d", "--depth", default=1, help="depth of the transformer model")
parser.add_argument("-g", "--max-nodes", default=80, help="maximum number of nodes in a generated graph")
parser.add_argument("-x", "--max-len", default=80, help="maximum number of actions per trajectory")
parser.add_argument("-r", "--random-action-template", default=2, help="index of the random action config to use (see code)")

# training
parser.add_argument("-l", "--loss-fn", default="tb-uniform", help="loss function for training (e.g. TB + uniform backward policy)")
parser.add_argument("-v", "--loss-arg", default=1)
parser.add_argument("-m", "--batch-size", default=32)
parser.add_argument("-p", "--num-precomputed", default=16, help="number of trajectories from precomputed, fully connected graphs")
parser.add_argument("-a", "--learning-rate", default=0.00001)
parser.add_argument("-n", "--max-update-norm", default=100)
parser.add_argument("-e", "--num-batches", default=5_000)

args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

configs = {
    "tb-uniform": (get_tb_loss_uniform, {"parameterise_backward": False}),
    "tb-add-node-mult": (lambda *args, **kwargs: get_tb_loss_add_node_mult(*args, n=args.loss_arg, **kwargs), {"parameterise_backward": False}),
    "tb-tlm": (get_tb_loss_tlm, {"parameterise_backward": True})
}

get_loss, config = configs[args.loss_fn]
parameterise_backward = config["parameterise_backward"]

base_model = torch.compile(GraphTransformer(dim=args.num_features, depth=args.depth, edge_dim=args.num_features, with_feedforwards=True, gated_residual=True, rel_pos_emb=False)).to(args.device)

fwd_stop_model = torch.compile(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
fwd_node_model = torch.compile(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
fwd_edge_model = torch.compile(nn.Sequential(nn.Linear(args.num_features*3, args.num_features*3*2), nn.LeakyReLU(), nn.Linear(args.num_features*3*2, 1))).to(args.device)
fwd_models = [fwd_stop_model, fwd_node_model, fwd_edge_model]

if parameterise_backward:
    bck_stop_model = torch.compile(nn.Sequential(nn.Linear(args.num_features, args.num_featuresargs.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
    bck_node_model = torch.compile(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
    bck_edge_model = torch.compile(nn.Sequential(nn.Linear(args.num_features*3, args.num_features*3*2), nn.LeakyReLU(), nn.Linear(args.num_features*3*2, 1))).to(args.device)
    bck_models = [bck_stop_model, bck_node_model, bck_edge_model]
else:
    bck_models = []

log_z_model = torch.compile(nn.Linear(1, 1, bias=False)).to(args.device)

random_configs = [(0, 0, 0),
                  (0.5, 0, 0.5),
                  (1, 0.99, 0.1)]
random_prob, random_prob_decay, random_prob_min = random_configs[args.random_action_template]

main_optimiser = torch.optim.Adam(base_model.parameters(), lr=args.learning_rate, weight_decay=1e-6)  # TODO: check regularisation is ok (prev: 0)
fwd_optimiser = torch.optim.Adam(itertools.chain(*(i.parameters() for i in fwd_models)), lr=args.learning_rate, weight_decay=1e-6)
log_z_optimiser = torch.optim.Adam(log_z_model.parameters(), lr=args.learning_rate, weight_decay=1e-6)

main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(main_optimiser, T_max=args.num_batches)
fwd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fwd_optimiser, T_max=args.num_batches)
log_z_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(log_z_optimiser, T_max=args.num_batches)


if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    reward_fn_generator = get_reward_fn_generator(get_smoothed_log_reward, base=args.base)

    data_source = GFNSampler(base_model, *fwd_models, reward_fn_generator,
                             node_features=args.num_features, edge_features=args.num_featuresargs.num_features,
                             random_action_prob=random_prob, max_len=args.max_len, max_nodes=args.max_nodes, base=args.base,
                             batch_size=args.batch_size, num_precomputed=args.num_precomputed,
                             device=args.device)
    data_loader = torch.utils.data.DataLoader(data_source, batch_size=None)

    sum_loss = mean_log_reward = mean_connected_prop = mean_num_nodes = 0

    for it, (jagged_trajs, log_rewards) in zip(range(args.num_batches), data_loader):

        loss, metrics = get_loss(jagged_trajs, log_rewards, base_model, log_z_model, *fwd_models, *bck_models, device=args.device)
        loss.backward()

        sq_norm = 0
        for m in (base_model, *fwd_models, log_z_model):  # TODO: clip norm of the combined updates
            sq_norm += min(args.max_norm, torch.nn.utils.clip_grad_norm_(m.parameters(), args.max_norm).item()) ** 2

        main_optimiser.step()
        main_optimiser.zero_grad()
        fwd_optimiser.step()
        fwd_optimiser.zero_grad()
        log_z_optimiser.step()
        log_z_optimiser.zero_grad()

        main_scheduler.step()
        fwd_scheduler.step()
        log_z_scheduler.step()

        sum_loss += loss.detach() / args.cycle_len
        mean_log_reward += metrics["mean_log_reward"] / args.cycle_len
        mean_connected_prop += metrics["connected_prop"] / args.cycle_len
        mean_num_nodes += metrics["mean_num_nodes"] / args.cycle_len

        with torch.no_grad():

            if (it+1)%args.cycle_len == 0:

                test_mean_log_reward = test_mean_connected_prop = 0
                test_node_counts = []

                graphs = data_source.generate_graphs(args.num_test_graphs)
                for i, (nodes, edges, masks) in enumerate(graphs):
                    num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
                    num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))

                    test_mean_log_reward += get_smoothed_log_reward(nodes.reshape((1, *nodes.shape)), edges.reshape((1, *edges.shape)), alpha=1_000_000).item()
                    test_mean_connected_prop += float(num_edges == num_nodes**2)
                    test_node_counts.append(num_nodes.item())

                    if args.save:
                        np.save(f"results/batches/nodes_{it}_{i}.npy", nodes.to("cpu").numpy())
                        np.save(f"results/batches/edges_{it}_{i}.npy", edges.to("cpu").numpy())
                        np.save(f"results/batches/masks_{it}_{i}.npy", edges.to("cpu").numpy())

                test_mean_log_reward /= args.num_test_graphs if args.num_test_graphs != 0 else 1
                test_mean_connected_prop /= args.num_test_graphs if args.num_test_graphs != 0 else 1

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

                data_source.random_action_prob = max(random_prob_min, data_source.random_action_prob * random_prob_decay)

                if args.save:
                    names = ("stop_model", "node_model", "edge_model")
                    for m, f in zip([base_model, *fwd_models, *bck_models, log_z_model],
                                    ["base_model", *("fwd_" + n for n in names), *("bck_" + n for n in names), "log_z_model"]):
                        torch.save(m.state_dict(), f"results/models/{f}_{it}.pt")

    print("done.")
