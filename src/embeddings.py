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


parser = argparse.ArgumentParser()

parser.add_argument("-s", "--seed", type=int, default=1)
parser.add_argument("-t", "--save-template", action="store_true", default=False, help="generate results/s/template.npy")
parser.add_argument("-n", "--num-features", type=int, default=25, help="number of features for inputs in the template")
parser.add_argument("-l", "--template-length", type=int, default=1024, help="approx. number of entries in the template")
parser.add_argument("-b", "--batch-size", type=int, default=32, help="size of each template batch")
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
    from data_source import GFNSampler, get_reward_fn_generator, get_smoothed_log_reward

    INTERNAL_BATCH_SIZE = 32
    NUM_INTERNAL_BATCHES = round(args.template_length / INTERNAL_BATCH_SIZE)
    BATCH_ARRGEGATION = round(args.batch_size / INTERNAL_BATCH_SIZE)

    base_model = GraphTransformer(dim=args.num_features, depth=1, edge_dim=args.num_features, with_feedforwards=True, gated_residual=True, rel_pos_emb=False)
    fwd_models = [nn.Linear(args.num_features, 1), nn.Linear(args.num_features, 1), nn.Linear(args.num_features*3, 1)]

    data_source = GFNSampler(base_model, *fwd_models, get_reward_fn_generator(get_smoothed_log_reward),
                             node_features=args.num_features, edge_features=args.num_features,
                             random_action_prob=1, adjust_random=4, max_len=80, max_nodes=8,
                             batch_size=INTERNAL_BATCH_SIZE, num_precomputed=0, device="cpu")
    data_loader = torch.utils.data.DataLoader(data_source, batch_size=None)

    outs = np.array([None]*ceil(NUM_INTERNAL_BATCHES / BATCH_ARRGEGATION), dtype=object)
    batch = []

    for it, (jagged_trajs, _log_rewards) in zip(range(NUM_INTERNAL_BATCHES), data_loader):

        for traj in jagged_trajs:

            idx = random.randint(0, len(traj) - 2)
            batch.append(traj[idx])

        if (it+1) % BATCH_ARRGEGATION == 0 or it + 1 == NUM_INTERNAL_BATCHES:

            # pad inputs to the same length (this is quite memory intensive)
            nodes = nn.utils.rnn.pad_sequence([n for (n, _e, _m), _a in batch], batch_first=True)
            edges = torch.stack([F.pad(e, (0, 0, 0, nodes.shape[1] - e.shape[1], 0, nodes.shape[1] - e.shape[0]), "constant", 0) for (_n, e, _m), _a in batch])
            masks = torch.stack([F.pad(m, (0, nodes.shape[1] - m.shape[0]), "constant", 0) for (_n, _e, m), _a in batch])
            outs[it // BATCH_ARRGEGATION] = (nodes, edges, masks)

            batch = []

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

    for i in range(99, 30_000, 100):
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

if args.show_graph or args.save_graph:  # TODO: we want to produce a surface from some of the points and a path over the surface from some of the others

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
