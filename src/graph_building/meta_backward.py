import subprocess
import numpy as np
import pygad
import torch


PARALLEL = 2
SEED = 1
N = 3
dp = {}

def fitness_func(ga_instance, weights, solution_idx):
    solution_idx += ga_instance.generations_completed * 10
    if tuple(weights.tolist()) in dp.keys():
        return dp[tuple(weights.tolist())]
    weights = torch.tensor(weights, dtype=torch.float32) ** N  # high N helps to focus on only a few weights
    torch.save(weights / torch.sum(weights), f"results/meta_weights_{solution_idx}.pt")  # this does not normalise: weights are not bounded since they can be negative
    _return_code = subprocess.call(
        f"python src/graph_building/main.py --reward-idx 2 --loss-fn meta --num-batches 5000 --meta-test {solution_idx} --seed {SEED} --cycle-len -1", shell=True
    )
    fitness = np.load(f"results/meta_fitness_{solution_idx}.npy")
    dp[tuple(weights.tolist())] = fitness
    return fitness

ga_instance = pygad.GA(
    num_generations=10,
    num_parents_mating=4,
    keep_elitism=6,
    fitness_func=fitness_func,
    sol_per_pop=10,
    num_genes=11,
    init_range_low=0,
    init_range_high=0.18,
    mutation_probability=0.18,
    random_mutation_min_val=-0.35,
    random_mutation_max_val=0.35,
    save_solutions=True,  # make sure saves what round and the fitness as well
    random_seed=SEED,
    parallel_processing=("process", PARALLEL) if PARALLEL > 1 else None
)

ga_instance.run()

solutions = torch.tensor(ga_instance.solutions) ** N
solutions = (solutions.T / torch.sum(solutions, axis=1)).T

np.save("results/m/solutions.npy", solutions.numpy())
np.save("results/m/results.npy", np.array([dp[tuple(s.tolist())] for s in solutions]))
