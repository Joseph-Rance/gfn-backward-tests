import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

SEED = 1
NUM_RANDOM = 8
MEANS = (0.2, 0.5)
NUM_UNIFORM = 2
NUM_TLM = 3


def save_diagram(fwd_embeddings, bck_embeddings, losses, colours):

    tsne = manifold.TSNE(n_components=1, random_state=SEED)
    fwd_vals = np.flatten(tsne.fit_transform(fwd_embeddings))
    bck_vals = np.flatten(tsne.fit_transform(bck_embeddings))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(fwd_vals, bck_vals, losses, c=colours)

    ax.set_xlabel("forward policy")
    ax.set_ylabel("backward policy")
    ax.set_zlabel("loss")

    # TODO: temp
    #plt.savefig("tsne_plot.png")
    plt.show()


fwd_embeddings = []
bck_embeddings = []
losses = []
colours = []

for seed in range(1, NUM_RANDOM+1):
    for mean in MEANS:

        _return_code = subprocess.call(
            f"PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform-rand --loss-arg-a {mean} --loss-arg-b {mean} --loss-arg-c {seed} --seed {seed} --test-template", shell=True
        )

        for it in range(4, 5004, 5):
            fwd_embeddings.append(np.load(f"results/embeddings/fwd_{it}.npy"))
            bck_embeddings.append(np.load(f"results/embeddings/bck_{it}.npy"))

        losses += np.load(f"results/losses.npy").tolist()
        colours += [0]*len(losses)

        np.save(f"results/s/fwd_embeddings.npy", np.array(fwd_embeddings))
        np.save(f"results/s/bck_embeddings.npy", np.array(bck_embeddings))
        np.save(f"results/s/losses.npy", np.array(losses))
        np.save(f"results/s/colours.npy", np.array(colours))

        save_diagram(np.array(fwd_embeddings), np.array(bck_embeddings), np.array(losses), np.array(colours))

for seed in range(1, NUM_RANDOM+1):
    for mean in MEANS:

        _return_code = subprocess.call(
            f"PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform-rand-var --loss-arg-a {mean} --loss-arg-b {mean} --seed {seed} --test-template", shell=True
        )

        for it in range(4, 5004, 5):
            fwd_embeddings.append(np.load(f"results/embeddings/fwd_{it}.npy"))
            bck_embeddings.append(np.load(f"results/embeddings/bck_{it}.npy"))

        losses += np.load(f"results/losses.npy").tolist()
        colours += [0]*len(losses)

        np.save(f"results/s/fwd_embeddings.npy", np.array(fwd_embeddings))
        np.save(f"results/s/bck_embeddings.npy", np.array(bck_embeddings))
        np.save(f"results/s/losses.npy", np.array(losses))
        np.save(f"results/s/colours.npy", np.array(colours))

        save_diagram(np.array(fwd_embeddings), np.array(bck_embeddings), np.array(losses), np.array(colours))

for seed in range(1, NUM_UNIFORM+1):

    _return_code = subprocess.call(
        f"PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform --seed {seed} --test-template", shell=True
    )

    for it in range(4, 5004, 5):
        fwd_embeddings.append(np.load(f"results/embeddings/fwd_{it}.npy"))
        bck_embeddings.append(np.load(f"results/embeddings/bck_{it}.npy"))

    losses += np.load(f"results/losses.npy").tolist()
    colours += [seed] * len(losses)

    np.save(f"results/s/fwd_embeddings.npy", np.array(fwd_embeddings))
    np.save(f"results/s/bck_embeddings.npy", np.array(bck_embeddings))
    np.save(f"results/s/losses.npy", np.array(losses))
    np.save(f"results/s/colours.npy", np.array(colours))

    save_diagram(np.array(fwd_embeddings), np.array(bck_embeddings), np.array(losses), np.array(colours))

for seed in range(1, NUM_TLM+1):

    _return_code = subprocess.call(
        f"PYTHONHASHSEED=0 python src/main.py --loss-fn tb-tlm --num-batches 10000 --seed {seed} --test-template", shell=True
    )

    for it in range(4, 5004, 5):
        fwd_embeddings.append(np.load(f"results/embeddings/fwd_{it}.npy"))
        bck_embeddings.append(np.load(f"results/embeddings/bck_{it}.npy"))

    losses += np.load(f"results/losses.npy").tolist()
    colours += [seed+NUM_UNIFORM] * len(losses)

    np.save(f"results/s/fwd_embeddings.npy", np.array(fwd_embeddings))
    np.save(f"results/s/bck_embeddings.npy", np.array(bck_embeddings))
    np.save(f"results/s/losses.npy", np.array(losses))
    np.save(f"results/s/colours.npy", np.array(colours))

    save_diagram(np.array(fwd_embeddings), np.array(bck_embeddings), np.array(losses), np.array(colours))

for seed in range(1, NUM_TLM+1):

    _return_code = subprocess.call(
        f"PYTHONHASHSEED=0 python src/main.py --loss-fn tb-smoothed-tlm --loss-arg-a 0.5 --seed {seed} --test-template", shell=True
    )

    for it in range(4, 5004, 5):
        fwd_embeddings.append(np.load(f"results/embeddings/fwd_{it}.npy"))
        bck_embeddings.append(np.load(f"results/embeddings/bck_{it}.npy"))

    losses += np.load(f"results/losses.npy").tolist()
    colours += [seed+NUM_TLM+NUM_UNIFORM] * len(losses)

    np.save(f"results/s/fwd_embeddings.npy", np.array(fwd_embeddings))
    np.save(f"results/s/bck_embeddings.npy", np.array(bck_embeddings))
    np.save(f"results/s/losses.npy", np.array(losses))
    np.save(f"results/s/colours.npy", np.array(colours))

    save_diagram(np.array(fwd_embeddings), np.array(bck_embeddings), np.array(losses), np.array(colours))

print("DONE.")
