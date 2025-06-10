import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch_scatter import scatter
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-g", "--grid-size", type=int, default=7)
parser.add_argument("-m", "--right-multiplier", type=float, default=1.)
parser.add_argument("-n", "--noise", type=float, default=0.0)

args = parser.parse_args()

BATCH_SIZE = 256
NUM_BATCHES = 10_000

rand_prob = 0.5

reward_dist = np.array([[j*(args.grid_size-1-j) + i*(args.grid_size-1-i) for j in range(args.grid_size)] for i in range(args.grid_size)]) ** 5
reward_dist = np.maximum(reward_dist / np.max(reward_dist), 0.00001)

tru_dist = reward_dist / np.sum(reward_dist)

#model = nn.Sequential(nn.Linear(2, 10), nn.LeakyReLU(), nn.Linear(10, 3)).to("cuda")
model = nn.Linear(2, 3, device="cuda")
bck_model = nn.Linear(2, 2, device="cuda")  # we never predict stop backwards because it can be directly computed (why not consider stop action to transition to a new state and therefore do this on other tasks?)
log_z_model = nn.Linear(1, 1, bias=False, device="cuda")

fwd_optimiser = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
bck_optimiser = torch.optim.Adam(bck_model.parameters(), lr=1e-2, weight_decay=1e-6)
log_z_optimiser = torch.optim.Adam(log_z_model.parameters(), lr=1e-1, weight_decay=1e-6)

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

                if s[0] > (1 - 1 / 2 / (args.grid_size - 1)):
                    action_probs[act_idx][0] = -inf_val

                if s[1] > (1 - 1 / 2 / (args.grid_size - 1)):
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
                    states[i][0] += 1 / (args.grid_size - 1)
                elif a == 1:
                    states[i][1] += 1 / (args.grid_size - 1)
                else:
                    assert a == 2
                    done[i] = True

    assert all([t[-1][1] == 2 for t in jagged_trajs])

    final_states = [(round(t[-1][0][0] * (args.grid_size - 1)), round(t[-1][0][1] * (args.grid_size - 1))) for t in jagged_trajs]

    # compute divergence between generated and true distribution
    gen_dist = np.array([[0 for __ in range(args.grid_size)] for __ in range(args.grid_size)])
    for s in final_states:
        gen_dist[s] += 1
    gen_dist = gen_dist / np.sum(gen_dist)
    eta = 0.0000000001
    kl = np.sum(np.maximum(eta, tru_dist) * np.log(np.maximum(eta, tru_dist) / np.maximum(eta, gen_dist))).item()

    # update model (clip between 0 and 2r for full experiment)
    log_rewards = torch.log(torch.clamp(torch.tensor([reward_dist[s] for s in final_states], device="cuda") + torch.normal(0, args.noise, size=(len(final_states),), device="cuda"), min=0.0001, max=2))

    traj_lens = torch.tensor([len(t) for t in jagged_trajs])
    trajs = [s_a for traj in jagged_trajs for s_a in traj]

    visit_count = np.array([[0 for __ in range(args.grid_size)] for __ in range(args.grid_size)])
    for s, _a in trajs:
        visit_count[round(s[0] * (args.grid_size - 1)), round(s[1] * (args.grid_size - 1))] += 1
    visit_count = visit_count / np.sum(visit_count)

    fwd_action_probs = model(torch.tensor([s for s, _a in trajs], device="cuda"))
    corr_fwd_action_probs = fwd_action_probs - torch.max(fwd_action_probs, dim=1).values.reshape((-1, 1))
    softmax_fwd_action_probs = corr_fwd_action_probs - torch.logsumexp(corr_fwd_action_probs, dim=1).reshape((-1, 1))
    log_p_f = softmax_fwd_action_probs[list(range(len(softmax_fwd_action_probs))), [a for _s, a in trajs]]
    
    bck_states = torch.tensor([s for s, _a in trajs], device="cuda")
    bck_states = torch.roll(bck_states, -1, 0)

    bck_action_probs = bck_model(bck_states)
    bck_fwd_action_probs = bck_action_probs - torch.max(bck_action_probs, dim=1).values.reshape((-1, 1))
    softmax_bck_action_probs = bck_fwd_action_probs - torch.logsumexp(bck_fwd_action_probs, dim=1).reshape((-1, 1))
    parameterised_log_p_b = softmax_bck_action_probs[list(range(len(softmax_bck_action_probs))), [0 if a == 2 else a for _s, a in trajs]]
    parameterised_log_p_b[np.cumsum(traj_lens) - 1] = 0

    #backward_probs = [(args.right_multiplier if a == 1 else 1) / max((0. if s[0] == 0 else 1.) + (0. if s[1] == 0 else args.right_multiplier), 0.0001)
    #                    for (s, _a), (_s, a) in zip(trajs, [trajs[-1]] + trajs[:-1])]
    #uniform_log_p_b = torch.log(torch.tensor(backward_probs))

    #uniform_log_p_b = torch.roll(uniform_log_p_b, -1, 0)
    #uniform_log_p_b[torch.cumsum(traj_lens, 0) - 1] = 0

    log_p_b = parameterised_log_p_b

    batch_idx = torch.arange(len(traj_lens), device="cuda").repeat_interleave(traj_lens.to("cuda"))
    traj_log_p_f = scatter(log_p_f, batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = scatter(log_p_b.to("cuda"), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

    bck_loss = -traj_log_p_b.mean()

    traj_log_p_b = traj_log_p_b.detach()

    log_z = log_z_model(torch.tensor([[1.]], device="cuda"))

    traj_diffs = (log_z + traj_log_p_f) - (log_rewards + traj_log_p_b)
    tb_loss = (traj_diffs * traj_diffs).mean()

    combined_loss = tb_loss + bck_loss

    combined_loss.backward()

    fwd_optimiser.step()
    fwd_optimiser.zero_grad()
    bck_optimiser.step()
    bck_optimiser.zero_grad()
    log_z_optimiser.step()
    log_z_optimiser.zero_grad()

    rand_prob *= 0.999

    if (it+1) % 100 == 0:

        print(f"it: {it:>7}, rand: {rand_prob:.3f}, log_z: {log_z.item():6.3f}, loss: {tb_loss.item():6.3f}, mean length: {traj_lens.to(float).mean().item():6.3f}, mean reward: {torch.exp(log_rewards).mean().item():.3f} divergence: {kl:7.3f}, gen_dist_marginal_0: {' '.join([f'{i:.3f}' for i in np.sum(gen_dist, axis=1)])}, gen_dist_marginal_1: {' '.join([f'{i:.3f}' for i in np.sum(gen_dist, axis=0).tolist()])}")

        with torch.no_grad():
            states = torch.tensor([[(i / (args.grid_size - 1), j / (args.grid_size - 1)) for j in range(args.grid_size)] for i in range(args.grid_size)], device="cuda").reshape(-1, 2)
            fwd_action_probs = model(states)

            for i, s in enumerate(states):
                if s[0] > (1 - 1 / 2 / (args.grid_size - 1)):
                    fwd_action_probs[i][0] = -inf_val
                if s[1] > (1 - 1 / 2 / (args.grid_size - 1)):
                    fwd_action_probs[i][1] = -inf_val

            unnorm_softmax_action_probs = torch.exp(fwd_action_probs - torch.max(fwd_action_probs, dim=1).values.reshape((-1, 1)))
            softmax_action_probs = unnorm_softmax_action_probs / torch.sum(unnorm_softmax_action_probs, dim=1).reshape((-1, 1))

            down_probs = softmax_action_probs[:, 0].cpu().detach().numpy().reshape((7, 7))
            right_probs = softmax_action_probs[:, 1].cpu().detach().numpy().reshape((7, 7))
            stop_probs = softmax_action_probs[:, 2].cpu().detach().numpy().reshape((7, 7))

        grid = np.zeros((13, 13))
        flows = np.zeros((7, 7))

        flows[0, 0] = log_z.exp().item()

        for i in range(13):
            for j in range(13):
                if i % 2 == 0 and j % 2 == 0:  # node
                    from_left = 0 if j == 0 else grid[i, j-1]
                    from_up = 0 if i == 0 else grid[i-1, j]
                    flows[i//2, j//2] += from_left + from_up
                    grid[i, j] = flows[i//2, j//2] * stop_probs[i//2, j//2]
                elif i % 2 == 0 and j % 2 != 0:  # right edge
                    grid[i, j] = flows[i//2, j//2] * right_probs[i//2, j//2]
                elif i % 2 != 0 and j % 2 == 0:  # down edge
                    grid[i, j] = flows[i//2, j//2] * down_probs[i//2, j//2]

        plt.imshow(grid, cmap="Grays")
        plt.savefig(f"biased_grid_{it}.png")

        #plt.imshow(visit_count, cmap="Grays")
        #plt.savefig(f"biased_visit_{it}.png")

        plt.imshow(gen_dist, cmap="Grays")
        plt.savefig(f"biased_dist_{it}.png")

        #print(traj_lens.tolist()[:15])
        #print(traj_log_p_f.tolist()[:15])
        #print(log_rewards.tolist()[:15])
        #print(traj_log_p_b.tolist()[:15])

print(f"grid:\n{grid}")

END_BS = 2048

with torch.no_grad():

    states = torch.tensor([(0., 0.)] * END_BS)
    done = torch.tensor([False] * END_BS)

    while not all(done):

        action_probs = model(states[~done].to("cuda"))
        act_idx = 0

        for i, s in enumerate(states):
            if done[i]:
                continue
            if s[0] > (1 - 1 / 2 / (args.grid_size - 1)):
                action_probs[act_idx][0] = -inf_val
            if s[1] > (1 - 1 / 2 / (args.grid_size - 1)):
                action_probs[act_idx][1] = -inf_val
            act_idx += 1

        unnorm_softmax_action_probs = torch.exp(action_probs - torch.max(action_probs, dim=1).values.reshape((-1, 1)))
        softmax_action_probs = unnorm_softmax_action_probs / torch.sum(unnorm_softmax_action_probs, dim=1).reshape((-1, 1))
        acts = torch.multinomial(softmax_action_probs, 1).reshape((-1,)).tolist()

        for i, s in enumerate(states):
            if done[i]:
                continue
            a = acts.pop(0)
            if a == 0:
                states[i][0] += 1 / (args.grid_size - 1)
            elif a == 1:
                states[i][1] += 1 / (args.grid_size - 1)
            else:
                assert a == 2
                done[i] = True

fmt_states = [(round(s[0].item() * (args.grid_size - 1)), round(s[1].item() * (args.grid_size - 1))) for s in states]
gen_dist = np.array([[0 for __ in range(args.grid_size)] for __ in range(args.grid_size)])
for s in fmt_states:
    gen_dist[s] += 1
gen_dist = gen_dist / np.sum(gen_dist)

print(f"mean loc: {states.mean(dim=0)}\ngenerated distribution: {gen_dist}")
