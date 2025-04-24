import collections
import itertools
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from graph_transformer_pytorch import GraphTransformer

from data_source import GFNSampler, get_reward_fn_generator, get_smoothed_log_reward, get_uncertain_smoothed_log_reward
from gfn import (
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
    get_tb_loss_smooth_tlm
)


parser = argparse.ArgumentParser()

# general
parser.add_argument("-s", "--seed", default=1)
parser.add_argument("-w", "--device", default="cuda", help="generally 'cuda' or 'cpu'")
parser.add_argument("-o", "--save", default=False, help="whether to save outputs to a file")
parser.add_argument("-c", "--cycle-len", default=5, help="how often to log/checkpoint (number of batches)")
parser.add_argument("-t", "--num-test-graphs", default=0, help="number of graphs to generate for estimating metrics")

# env
parser.add_argument("-b", "--base", default=0.8, help="base for exponent used in reward calculation")
parser.add_argument("-r", "--reward_idx", default=0, help="index of reward function to use")

# model
parser.add_argument("-f", "--num-features", default=10, help="number of features used to represent each node/edge (min 2)")
parser.add_argument("-d", "--depth", default=1, help="depth of the transformer model")
parser.add_argument("-g", "--max-nodes", default=8, help="maximum number of nodes in a generated graph")
parser.add_argument("-k", "--max-len", default=80, help="maximum number of actions per trajectory")
parser.add_argument("-q", "--random-action-template", default=2, help="index of the random action config to use (see code)")
parser.add_argument("-z", "--log-z", default=None, help="constant value of log(z) to use (learnt if None)")

# training
parser.add_argument("-l", "--loss-fn", default="tb-uniform", help="loss function for training (e.g. TB + uniform backward policy)")
parser.add_argument("-v", "--loss-arg-a", default=1)
parser.add_argument("-u", "--loss-arg-b", default=1)
parser.add_argument("-m", "--batch-size", default=32)
parser.add_argument("-p", "--num-precomputed", default=16, help="number of trajectories from precomputed, fully connected graphs")
parser.add_argument("-i", "--edges-first", default=False, help="whether to add edges before nodes in precomputed trajectories")
parser.add_argument("-a", "--learning-rate", default=0.00001)
parser.add_argument("-n", "--max-update-norm", default=100)
parser.add_argument("-e", "--num-batches", default=5_000)

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
        lambda *args, **kwargs: get_tb_loss_adjusted_uniform(*args, base=args.base, **kwargs),
        {"parameterise_backward": False}
    ),
    "tb-uniform-add-node": (  # uniform backward policy with an added bias towards adding nodes
        lambda *args, **kwargs: get_tb_loss_add_node_mult(*args, n=args.loss_arg_a, **kwargs),
        {"parameterise_backward": False}
    ),
    "tb-uniform-const": (  # uniform backward policy with a constant (unnormalised) backward probability
        lambda *args, **kwargs: get_tb_loss_const(*args, val=args.loss_arg_a, **kwargs),
        {"parameterise_backward": False}
    ),
    "tb-uniform-rand": (  # uniform backward probabilities randomly perturbed by noised sampled from a uniform distribution
        lambda *args, **kwargs: get_tb_loss_rand_const(*args, mean=args.loss_arg_a, std=args.loss_arg_b, **kwargs),
        {"parameterise_backward": False}
    ),
    "tb-uniform-rand-var": (  # backward probabilities randomly resampled from a normal distribution on each application
        lambda *args, **kwargs: get_tb_loss_rand_var(*args, mean=args.loss_arg_a, std=args.loss_arg_b, **kwargs),
        {"parameterise_backward": False}
    ),
    "tb-aligned": (  # aligned to handmade backward policy
        lambda *args, **kwargs: get_tb_loss_aligned(*args, base=args.base, correct_val=args.loss_arg_a, incorrect_val=args.loss_arg_b, **kwargs),
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
        lambda *args, **kwargs: get_tb_loss_biased_tlm(*args, multiplier=args.loss_arg_a, ns=[args.loss_arg_b], **kwargs),
        {"parameterise_backward": True}
    ),
    "tb-smoothed-tlm": (  # TLM / pessimistic mixed with a uniform distribution
        lambda *args, **kwargs: get_tb_loss_smooth_tlm(*args, a=args.loss_arg_a, **kwargs),  # maybe cosine annealing curve from 1 to 0 over time?
        {"parameterise_backward": True}
    )
}

get_loss, config = configs[args.loss_fn]
parameterise_backward = config["parameterise_backward"]

reward_fns = [get_smoothed_log_reward, get_uncertain_smoothed_log_reward]
reward_fn = reward_fns[args.reward_idx]

#compile = lambxa x: torch.compile(x)
compile = lambda x: x

base_model = compile(GraphTransformer(dim=args.num_features, depth=args.depth, edge_dim=args.num_features, with_feedforwards=True, gated_residual=True, rel_pos_emb=False)).to(args.device)

fwd_stop_model = compile(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
fwd_node_model = compile(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
fwd_edge_model = compile(nn.Sequential(nn.Linear(args.num_features*3, args.num_features*3*2), nn.LeakyReLU(), nn.Linear(args.num_features*3*2, 1))).to(args.device)
fwd_models = [fwd_stop_model, fwd_node_model, fwd_edge_model]

if parameterise_backward:
    bck_stop_model = compile(nn.Sequential(nn.Linear(args.num_features, args.num_featuresargs.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
    bck_node_model = compile(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device)
    bck_edge_model = compile(nn.Sequential(nn.Linear(args.num_features*3, args.num_features*3*2), nn.LeakyReLU(), nn.Linear(args.num_features*3*2, 1))).to(args.device)
    bck_models = [bck_stop_model, bck_node_model, bck_edge_model]
else:
    bck_models = []

if args.loss_fn == "tb-max-ent":
    assert parameterise_backward
    bck_models.append(compile(nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1))).to(args.device))

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

    sum_loss = mean_log_reward = mean_connected_prop = mean_num_nodes = 0

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

                    test_mean_log_reward += reward_fn(nodes.reshape((1, *nodes.shape)), edges.reshape((1, *edges.shape)), alpha=1_000_000).item()
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
                    f"norm: {norm:6.3f}; " \
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
