import numpy as np
import matplotlib.pyplot as plt

n_vals = [0.010, 0.012, 0.019, 0.040, 0.107, 0.321, 1.000, 3.156, 10.000]

means, props = [], []
for n_val in n_vals:
    means_n, props_n = [], []
    for run in [0, 1, 2]:
        res = np.load(f"results_{run}/counts_{int(n_val*1_000)}.npy")
        total_graphs = sum([i[1] for i in res])
        mean_nodes = sum([float(np.prod(i)/total_graphs) for i in res])
        means_n.append(mean_nodes)
        proportions = float(np.load(f"results_{run}/prop_{int(n_val*1_000)}.npy")[0])
        props_n.append(proportions)
    means.append(means_n)
    props.append(props_n)

means = np.array(means)
props = np.array(props)

mean_means = np.mean(means, axis=1)
mean_props = np.mean(props, axis=1)

plt.plot(n_vals, mean_means)
#plt.plot(n_vals, mean_props)
plt.xscale("log")
plt.yscale("log")
plt.savefig("out.png")