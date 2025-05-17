import subprocess
import numpy as np
import pygad
import torch


SEED = 1
N = 3
dp = {}

def fitness_func(_ga_instance, weights, _solution_idx):
    if tuple(weights.tolist()) in dp.keys():
        return dp[tuple(weights.tolist())]
    weights = torch.tensor(weights, dtype=torch.float32) ** N  # high N helps to focus on only a few weights
    torch.save(weights / torch.sum(weights), "results/meta_weights.pt")  # this does not normalise: weights are not bounded since they can be negative
    _return_code = subprocess.call(
        f"python src/graph_building/main.py --reward-idx 2 --loss-fn meta --num-batches 5000 --meta-test --seed {SEED} --save", shell=True
    )
    fitness = np.load("results/meta_fitness.npy")
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
    random_seed=SEED
)

ga_instance.run()

solutions = torch.tensor(ga_instance.solutions) ** N
solutions = (solutions.T / torch.sum(solutions, axis=1)).T

np.save("results/m/solutions.npy", solutions.numpy())
np.save("results/m/results.npy", np.array([dp[tuple(s.tolist())] for s in solutions]))
