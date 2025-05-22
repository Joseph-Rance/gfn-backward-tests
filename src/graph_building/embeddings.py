from math import ceil
import random
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import torch
import torch.nn as nn
import torch.nn.functional as F

from gfn import trajs_to_tensors


parser = argparse.ArgumentParser()

parser.add_argument("-s", "--seed", type=int, default=1)
parser.add_argument("-d", "--device", type=str, default="cuda", help="usually 'cuda' or 'cpu'")
parser.add_argument("-t", "--save-template", action="store_true", default=False, help="generate results/s/template.npy")
parser.add_argument("-e", "--model-path", type=str, default=".", help="path to directory with files containing forward models (for importance sampling)")
parser.add_argument("-n", "--num-features", type=int, default=16, help="number of features for inputs in the template")
parser.add_argument("-k", "--depth", type=int, default=2, help="number of features for inputs in the template")
parser.add_argument("-l", "--template-length", type=int, default=1024, help="approx. number of entries in the template")
parser.add_argument("-b", "--batch-size", type=int, default=32, help="size of each template batch")
parser.add_argument("-a", "--random-action-prob", type=float, default=0.25, help="probability of random actions when sampling trajectories")
parser.add_argument("-r", "--results-dir", type=str, default="results", help="directory to get results from")
parser.add_argument("-i", "--init-embeddings", action="store_true", default=False, help="generate files in results/s")
parser.add_argument("-m", "--merge-embeddings", action="store_true", default=False, help="merge results/embeddings into results/s")
parser.add_argument("-c", "--run-colour", type=int, default=0, help="colour to assign to the run being merged in (-1 to get colour from results/colours.npy)")
parser.add_argument("-g", "--process-data", action="store_true", default=False, help="save the data for a graph")
parser.add_argument("-p", "--show-graph", action="store_true", default=False, help="show a matplotlib graph in a separate window")
parser.add_argument("-w", "--save-graph", action="store_true", default=False, help="save a matplotlib graph as a file")
parser.add_argument("-u", "--template-filename", type=str, default="s/template.npy", help="filename of template starting from results-dir")
parser.add_argument("-v", "--graph-filename", type=str, default="s/loss_surface.png", help="filename of output png starting from results-dir")

args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


if args.save_template:

    from graph_transformer_pytorch import GraphTransformer
    from data_source import GFNSampler, get_reward_fn_generator, get_uniform_counting_log_reward

    NUM_BATCHES = round(args.template_length / args.batch_size)

    base_model = GraphTransformer(dim=args.num_features, depth=args.depth, edge_dim=args.num_features, with_feedforwards=True, gated_residual=True, rel_pos_emb=False).to(args.device)
    fwd_stop_model = nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1)).to(args.device)
    fwd_node_model = nn.Sequential(nn.Linear(args.num_features, args.num_features*2), nn.LeakyReLU(), nn.Linear(args.num_features*2, 1)).to(args.device)
    fwd_edge_model = nn.Sequential(nn.Linear(args.num_features*3, args.num_features*3*2), nn.LeakyReLU(), nn.Linear(args.num_features*3*2, 1)).to(args.device)

    base_model.load_state_dict(torch.load(f"{args.model_path}/base_model.pt", weights_only=True))
    fwd_stop_model.load_state_dict(torch.load(f"{args.model_path}/fwd_stop_model.pt", weights_only=True))
    fwd_node_model.load_state_dict(torch.load(f"{args.model_path}/fwd_node_model.pt", weights_only=True))
    fwd_edge_model.load_state_dict(torch.load(f"{args.model_path}/fwd_edge_model.pt", weights_only=True))

    fwd_models = [fwd_stop_model, fwd_node_model, fwd_edge_model]

    data_source = GFNSampler(base_model, *fwd_models, get_reward_fn_generator(get_uniform_counting_log_reward),
                             node_features=args.num_features, edge_features=args.num_features,
                             random_action_prob=args.random_action_prob, max_len=72, max_nodes=8,
                             batch_size=args.batch_size, num_precomputed=0, device=args.device)
    data_loader = torch.utils.data.DataLoader(data_source, batch_size=None)

    outs = np.array([None]*NUM_BATCHES, dtype=object)

    for it, (jagged_trajs, _log_rewards) in zip(range(NUM_BATCHES), data_loader):

        traj_lens = torch.tensor([len(t) for t in jagged_trajs])
        trajs = [s for traj in jagged_trajs for s in traj]
        nodes, edges, masks, actions = trajs_to_tensors(trajs)

        outs[it] = (nodes, edges, masks, actions, traj_lens)

    np.save(args.results_dir + "/" + args.template_filename, outs, allow_pickle=True)

if args.init_embeddings:

    np.save(f"{args.results_dir}/s/fwd_embeddings.npy", np.zeros((0,)))
    np.save(f"{args.results_dir}/s/bck_embeddings.npy", np.zeros((0,)))
    np.save(f"{args.results_dir}/s/losses.npy", np.zeros((0,)))
    np.save(f"{args.results_dir}/s/colours.npy", np.zeros((0,)))

if args.merge_embeddings:

    assert os.path.isfile(f"{args.results_dir}/s/fwd_embeddings.npy") \
       and os.path.isfile(f"{args.results_dir}/s/bck_embeddings.npy") \
       and os.path.isfile(f"{args.results_dir}/s/losses.npy") \
       and os.path.isfile(f"{args.results_dir}/s/colours.npy")

    fwd_embeddings = [i for i in np.load(f"{args.results_dir}/s/fwd_embeddings.npy")]
    bck_embeddings = [i for i in np.load(f"{args.results_dir}/s/bck_embeddings.npy")]

    #for f in os.listdir(f"{args.results_dir}/embeddings"):
    # 
    #    if f[:4] not in ["fwd_", "bck_"]:
    #        continue

    for i in range(99, 10_000, 100):
        fwd_embeddings.append(np.load(f"{args.results_dir}/embeddings/fwd_{i}.npy"))
        bck_embeddings.append(np.load(f"{args.results_dir}/embeddings/bck_{i}.npy"))

    new_losses = np.load(f"{args.results_dir}/losses.npy")
    losses = np.concatenate((np.load(f"{args.results_dir}/s/losses.npy"), new_losses), axis=0)

    if args.run_colour == -1:
        colours = np.concatenate((np.load(f"{args.results_dir}/s/colours.npy"), np.load(f"{args.results_dir}/colours.npy")), axis=0)
    else:
        colours = np.concatenate((np.load(f"{args.results_dir}/s/colours.npy"), [args.run_colour] * len(new_losses)), axis=0)

    np.save(f"{args.results_dir}/s/fwd_embeddings.npy", np.array(fwd_embeddings))
    np.save(f"{args.results_dir}/s/bck_embeddings.npy", np.array(bck_embeddings))
    np.save(f"{args.results_dir}/s/losses.npy", losses)
    np.save(f"{args.results_dir}/s/colours.npy", colours)

if args.process_data:
    fwd_embeddings = np.load(f"{args.results_dir}/s/fwd_embeddings.npy")
    bck_embeddings = np.load(f"{args.results_dir}/s/bck_embeddings.npy")
    losses = np.load(f"{args.results_dir}/s/losses.npy")
    colours = np.load(f"{args.results_dir}/s/colours.npy")

    # TODO:
    # "It is highly recommended to use another dimensionality reduction method (e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the number of dimensions to a reasonable amount (e.g. 50) if the number of features is very high. This will suppress some noise and speed up the computation of pairwise distances between samples. For more tips see Laurens van der Maaten’s FAQ [2].""

    tsne = manifold.TSNE(n_components=1, random_state=1)
    fwd_vals = tsne.fit_transform(fwd_embeddings).flatten()
    bck_vals = tsne.fit_transform(bck_embeddings).flatten()

    plt.plot(fwd_vals)
    plt.savefig("test_a.png")
    plt.clf()
    plt.plot(bck_vals)
    plt.savefig("test_b.png")
    plt.clf()

    data = np.stack((fwd_vals, bck_vals, losses, colours))
    np.save(f"{args.results_dir}/s/processed_data.npy", data)

if args.show_graph or args.save_graph:

    # should add to this graph to show a (smoothed) solid surface + (smoothed) line over the surface

    if not args.process_data:
        fwd_vals, bck_vals, losses, colours = np.load(f"{args.results_dir}/s/processed_data.npy")

    from math import log
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(fwd_vals, bck_vals, [log(x) for x in losses], c=colours)
    ax.set_xlabel("forward policy")
    ax.set_ylabel("backward policy")
    ax.set_zlabel("loss")

if args.save_graph:
    plt.savefig(args.results_dir + "/" + args.graph_filename)

if args.show_graph:
    plt.show()

print("done.")
