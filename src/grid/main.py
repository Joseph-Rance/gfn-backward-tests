import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch_scatter import scatter

GRID_SIZE = 7
BATCH_SIZE = 1024
NUM_BATCHES = 1_000_000

rand_prob = 0.5

reward_dist = np.array([[j*(GRID_SIZE-1-j) + i*(GRID_SIZE-1-i) for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]) ** 5
reward_dist = np.maximum(reward_dist / np.max(reward_dist), 0.00001)

tru_dist = reward_dist / np.sum(reward_dist)

model = nn.Sequential(nn.Linear(2, 10), nn.LeakyReLU(), nn.Linear(10, 3)).to("cuda")#nn.Linear(2, 3, device="cuda")
log_z_model = nn.Linear(1, 1, bias=False, device="cuda")

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
log_z_optimiser = torch.optim.Adam(log_z_model.parameters(), lr=1e-2, weight_decay=1e-6)

inf_val = 1_000_000

for it in range(NUM_BATCHES):

    with torch.no_grad():

        states = torch.tensor([(0., 0.)] * BATCH_SIZE)
        done = torch.tensor([False] * BATCH_SIZE)

        jagged_trajs = [[] for __ in range(BATCH_SIZE)]

        while not all(done):

            action_probs = model(states[~done].to("cuda"))
            act_idx = 0

            # apply masks
            for i, s in enumerate(states):

                if done[i]:
                    continue

                if s[0] > (1 - 1 / 2 / (GRID_SIZE - 1)):
                    action_probs[act_idx][0] = -inf_val

                if s[1] > (1 - 1 / 2 / (GRID_SIZE - 1)):
                    action_probs[act_idx][1] = -inf_val

                act_idx += 1

            unnorm_softmax_action_probs = torch.exp(action_probs - torch.max(action_probs, dim=1).values.reshape((-1, 1)))
            softmax_action_probs = unnorm_softmax_action_probs / torch.sum(unnorm_softmax_action_probs, dim=1).reshape((-1, 1))
            acts = torch.multinomial(softmax_action_probs, 1).reshape((-1,)).tolist()

            # update states
            for i, s in enumerate(states):

                if done[i]:
                    continue

                a = acts.pop(0)

                jagged_trajs[i].append((tuple(s.tolist()), a))

                if a == 0:
                    states[i][0] += 1 / (GRID_SIZE - 1)
                elif a == 1:
                    states[i][1] += 1 / (GRID_SIZE - 1)
                else:
                    assert a == 2
                    done[i] = True

    assert all([t[-1][1] == 2 for t in jagged_trajs])

    final_states = [(round(t[-1][0][0] * (GRID_SIZE - 1)), round(t[-1][0][1] * (GRID_SIZE - 1))) for t in jagged_trajs]

    # compute divergence between generated and true distribution
    gen_dist = np.array([[0 for __ in range(GRID_SIZE)] for __ in range(GRID_SIZE)])
    for s in final_states:
        gen_dist[s] += 1
    gen_dist = gen_dist / np.sum(gen_dist)
    eta = 0.0000000001
    kl = np.sum(np.maximum(eta, tru_dist) * np.log(np.maximum(eta, tru_dist) / np.maximum(eta, gen_dist))).item()

    # update model
    log_rewards = torch.log(torch.tensor([reward_dist[s] for s in final_states], device="cuda"))

    traj_lens = torch.tensor([len(t) for t in jagged_trajs])
    trajs = [s_a for traj in jagged_trajs for s_a in traj]

    fwd_action_probs = model(torch.tensor([s for s, _a in trajs], device="cuda"))
    corr_fwd_action_probs = fwd_action_probs - torch.max(fwd_action_probs, dim=1).values.reshape((-1, 1))
    softmax_fwd_action_probs = corr_fwd_action_probs - torch.logsumexp(corr_fwd_action_probs, dim=1).reshape((-1, 1))
    log_p_f = softmax_fwd_action_probs[list(range(len(softmax_fwd_action_probs))), [a for _s, a in trajs]]

    backward_actions = torch.tensor([(0. if s[0] == 0 else 1.) + (0. if s[1] == 0 else 2.) for s, _a in trajs])  # TODO: change from 2. ?
    uniform_log_p_b = torch.log(1 / torch.clamp(backward_actions, min=0.0001))

    uniform_log_p_b = torch.roll(uniform_log_p_b, -1, 0)
    uniform_log_p_b[torch.cumsum(traj_lens, 0) - 1] = 0

    log_p_b = uniform_log_p_b

    batch_idx = torch.arange(len(traj_lens), device="cuda").repeat_interleave(traj_lens.to("cuda"))
    traj_log_p_f = scatter(log_p_f, batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = scatter(log_p_b.to("cuda"), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

    traj_log_p_b = traj_log_p_b.detach()

    log_z = log_z_model(torch.tensor([[1.]], device="cuda"))

    traj_diffs = (log_z + traj_log_p_f) - (log_rewards + traj_log_p_b)
    tb_loss = (traj_diffs * traj_diffs).mean()

    tb_loss.backward()

    optimiser.step()
    optimiser.zero_grad()
    log_z_optimiser.step()
    log_z_optimiser.zero_grad()

    rand_prob *= 0.999

    if it % 20 == 0:
        print(f"it: {it:>7}, rand: {rand_prob:.3f}, log_z: {log_z.item():5.3f}, loss: {tb_loss.item():6.3f}, mean length: {traj_lens.to(float).mean().item():6.3f}, mean reward: {torch.exp(log_rewards).mean().item():.3f} divergence: {kl:7.3f}, gen_dist_marginal_0: {' '.join([f'{i:.3f}' for i in np.sum(gen_dist, axis=1)])}, gen_dist_marginal_1: {' '.join([f'{i:.3f}' for i in np.sum(gen_dist, axis=0).tolist()])}")
        plt.imshow(gen_dist)
        plt.savefig("current.png")
        plt.imshow(tru_dist - gen_dist)
        plt.savefig("diff.png")

        #print(traj_lens.tolist()[:15])
        #print(traj_log_p_f.tolist()[:15])
        #print(log_rewards.tolist()[:15])
        #print(traj_log_p_b.tolist()[:15])
