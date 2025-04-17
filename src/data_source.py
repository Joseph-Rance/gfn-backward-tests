from math import log
import copy
import numpy as np
import torch
from torch.utils.data import IterableDataset

from gfn import get_embeddings, get_action_probs


def get_smoothed_log_reward(nodes, edges, base=0.8, alpha=10):
    num_nodes = torch.sum(torch.sum(nodes, dim=2) > 0, dim=1)
    num_edges = torch.sum(edges[:, :, :, 0], dim=(1, 2))
    #return torch.logical_and(num_nodes == 2, num_edges == 0).long()  # (for testing)
    fully_connectedness = (num_edges - num_nodes**2) ** 2
    return torch.clamp(log(base) * num_nodes - alpha * fully_connectedness, min=-1_000)

def get_reward_fn_generator(reward_fn, base=0.8, alpha_start=0.75, alpha_change=1.02):
    alpha = alpha_start
    while True:
        yield lambda *args, **kwargs: reward_fn(*args, base=base, alpha=alpha, **kwargs)
        alpha *= alpha_change
        alpha = min(alpha, 1_000_000)  # TODO: WAS GOOD WITH MAX?


class GFNSampler(IterableDataset):

    def __init__(
        self,
        base_model,
        stop_model,
        node_model,
        edge_model,
        reward_fn_generator,
        node_features=10,
        edge_features=10,
        random_action_prob=0.5,
        max_len=500,
        max_nodes=10,
        batch_size=128,
        num_precomputed=64,
        max_precomputed_len=4,
        base=0.8,
        node_history_bounds=(0, 1),  # inclusive
        edge_history_bounds=(0, 1),  # inclusive
        masked_action_value=-80,
        action_prob_clip_bounds=(-75, 75),  # inclusive
        replay_buffer_length=1024,
        start_size=4,
        expand_factor=2,
        device="cuda"
    ):
        super(GFNSampler).__init__()

        self.base_model = base_model
        self.stop_model = stop_model
        self.node_model = node_model
        self.edge_model = edge_model
        self.reward_fn_iterator = iter(reward_fn_generator)
        self.node_features = node_features
        self.edge_features = edge_features
        self.random_action_prob = random_action_prob
        self.max_len = max_len
        self.max_nodes = max_nodes
        self.batch_size = batch_size
        self.num_precomputed = num_precomputed
        self.max_precomputed_len = max_precomputed_len
        self.base = base
        self.node_history_bounds = node_history_bounds
        self.edge_history_bounds = edge_history_bounds
        self.masked_action_value = masked_action_value
        self.action_prob_clip_bounds = action_prob_clip_bounds
        self.replay_buffer_length = replay_buffer_length
        self.start_size = start_size
        self.expand_factor = expand_factor
        self.device = device

        assert self.node_features == self.edge_features

        self.precomputed = []

        self.replay_buffer_trajs = np.empty((self.replay_buffer_length,), dtype=object)
        self.replay_buffer_log_rewards = torch.zeros((self.replay_buffer_length,))
        self.replay_end, self.replay_saturated = 0, False

    def __iter__(self):
        return self

    @torch.no_grad()
    def __next__(self):

        trajs = []
        log_rewards = torch.zeros((0,))

        if self.batch_size > self.num_precomputed:
            sampled_trajs, sampled_log_rewards = self.get_sampled()

            for i in range(len(sampled_trajs)):

                self.replay_buffer_trajs[self.replay_end] = sampled_trajs[i]
                self.replay_buffer_log_rewards[self.replay_end] = sampled_log_rewards[i]
                self.replay_end += 1

                if self.replay_end >= self.replay_buffer_length:
                    self.replay_saturated = True
                    self.replay_end = 0

            max_idx = self.replay_buffer_length if self.replay_saturated else self.replay_end
            idxs = torch.randint(0, max_idx, (len(sampled_trajs),))

            trajs += self.replay_buffer_trajs[idxs].tolist()
            log_rewards = torch.concatenate((log_rewards, self.replay_buffer_log_rewards[idxs]))

        if self.num_precomputed > 0:
            precomputed_trajs, precomputed_log_rewards = self.get_precomputed()
            trajs += precomputed_trajs
            log_rewards = torch.concatenate((log_rewards, precomputed_log_rewards))

        return trajs, log_rewards

    def get_sampled(self):

        self.base_model.eval(), self.stop_model.eval(), self.node_model.eval(), self.edge_model.eval()

        sample_size = self.batch_size - self.num_precomputed

        trajs = [[] for __ in range(sample_size)]

        nodes = torch.zeros((sample_size, self.start_size, self.node_features), device="cpu")
        edges = torch.zeros((sample_size, self.start_size, self.start_size, self.edge_features), device="cpu")
        masks = torch.zeros((sample_size, self.start_size), dtype=bool, device="cpu")

        nodes[:, 0, 0] = 1
        masks[:, 0] = 1

        done = torch.tensor([False] * sample_size)

        for __ in range(self.max_len):
            
            embeddings, structure = get_embeddings(
                self.base_model,
                nodes[~done],
                edges[~done],
                masks[~done],
                device=self.device
            )

            action_probs = get_action_probs(
                *embeddings,
                *structure,
                self.stop_model,
                self.node_model,
                self.edge_model, 
                random_action_prob=self.random_action_prob,
                apply_masks=True,
                max_nodes=self.max_nodes,
                masked_action_value=self.masked_action_value,
                action_prob_clip_bounds=self.action_prob_clip_bounds
            )

            assert action_probs.shape == (done.shape[0] - torch.sum(done), nodes.shape[1]**2 + 2)

            selected_action_idxs = torch.multinomial(torch.exp(action_probs), 1).reshape((-1,))

            j = 0
            for i in range(sample_size):
                if not done[i]:
                    trajs[i].append(((torch.clone(nodes[i].to("cpu")), torch.clone(edges[i].to("cpu")), torch.clone(masks[i].to("cpu"))),
                                      int(selected_action_idxs[j])))
                    j += 1

            (nodes, edges, masks), done = self.update_state(nodes, edges, masks, done, selected_action_idxs)

            if torch.all(done):
                break

        log_rewards = next(self.reward_fn_iterator)(nodes, edges)

        for t, __ in enumerate(trajs):
            if done[t]:  # for unfinished trajectories, we are just going to lose the last state
                trajs[t].append(copy.deepcopy(trajs[t][-1]))  # add padding state (will be ignored later)
                trajs[t][-1][0][0][:, -1] = 1

        log_rewards[~done] = -1_000

        return trajs, log_rewards

    def update_state(self, nodes, edges, masks, done, selected_action_idxs):

        sample_size = len(nodes)

        # update nodes
        nodes[:, :, 0] += torch.sum(nodes, dim=2) > 0  # increment existing nodes
        nodes = torch.clamp(nodes, min=self.node_history_bounds[0], max=self.node_history_bounds[1])  # clip node histories to control tree property

        prev_len = masks.shape[1]
        next_node = torch.sum(masks, dim=1)
        add_node = (selected_action_idxs == prev_len**2)

        if torch.any(add_node):

            # this setup is really quite wasteful
            if torch.max(next_node) == masks.shape[1]:  # expand dynamic array
                expansion_prop = masks.shape[1] * (self.expand_factor-1)
                nodes = torch.concatenate((nodes, torch.zeros((sample_size, expansion_prop, self.node_features), device="cpu")), dim=1)
                edges = torch.concatenate((edges, torch.zeros((sample_size, masks.shape[1], expansion_prop, self.edge_features), device="cpu")), dim=2)
                edges = torch.concatenate((edges, torch.zeros((sample_size, expansion_prop, masks.shape[1] + expansion_prop, self.edge_features), device="cpu")), dim=1)
                masks = torch.concatenate((masks, torch.zeros((sample_size, expansion_prop), dtype=bool, device="cpu")), dim=1)

            idxs = [i for i in range(sample_size) if not done[i]]
            nodes[idxs, next_node[idxs], 0] = add_node.float()
            masks[idxs, next_node[idxs]] = add_node

        # update edges
        edges[:, :, :, 1] += (torch.sum(edges, dim=3) > 0).float()  # increment existing nodes
        edges[:, :, :, 1] = torch.clamp(edges[:, :, :, 1], min=self.edge_history_bounds[0], max=self.edge_history_bounds[1])  # clip edge histories to control tree property

        j = 0
        for i in range(sample_size):  # too lazy to vectorise
            if done[i]:
                continue
            if selected_action_idxs[j] < prev_len**2:
                edges[i, selected_action_idxs[j] // prev_len, selected_action_idxs[j] % prev_len, (0, 1)] = 1
            j += 1

        # update done
        done[~done] = (selected_action_idxs == prev_len**2 + 1).to("cpu")

        return (nodes, edges, masks), done

    def get_precomputed(self):

        trajs = []

        num_nodes = torch.randint(1, self.max_precomputed_len + 1, (self.num_precomputed,), device="cpu")

        for n in num_nodes:
            trajs.append(self.get_all_precomputed()[int(n)-1])

        log_rewards = log(self.base) * num_nodes

        #log_rewards = torch.tensor([0] * len(num_nodes))  # (for testing)

        return trajs, log_rewards

    def get_precomputed(self, importance_sample=True, base_multiplier=1.15):

        trajs = []

        if importance_sample:
            p_base = 1 - (1-self.base / base_multiplier)
            manual_graphs = torch.rand((self.num_precomputed,), device="cpu")
            num_nodes = torch.zeros((self.num_precomputed,), device="cpu")

            i = 1
            a = 1 - p_base
            while torch.any(num_nodes == 0) and i < self.max_precomputed_len:
                num_nodes[torch.logical_and(manual_graphs < a, num_nodes == 0)] = i
                a += p_base**i * (1 - p_base)
                i += 1

            num_nodes[num_nodes == 0] = self.max_precomputed_len
        
        else:
            num_nodes = torch.randint(1, self.max_precomputed_len + 1, (self.num_precomputed,), device="cpu")

        for n in num_nodes:
            trajs.append(self.get_all_precomputed()[int(n)-1])

        log_rewards = log(self.base) * num_nodes

        #log_rewards = torch.tensor([0] * len(num_nodes))  # (for testing)

        return trajs, log_rewards

    def get_all_precomputed(self):
        
        if not self.precomputed:
        
            for n in range(1, self.max_precomputed_len+1):

                traj = []

                nodes = torch.zeros((1, self.start_size, self.node_features), device="cpu")
                edges = torch.zeros((1, self.start_size, self.start_size, self.edge_features), device="cpu")
                masks = torch.zeros((1, self.start_size), dtype=bool, device="cpu")

                nodes[0, 0, 0] = 1
                masks[0, 0] = 1

                actions = []

                size = self.start_size
                for n_i in range(int(n)-1):
                    actions += [size**2]
                    if n_i+1 == size:
                        size *= self.expand_factor

                for j in range(int(n)):
                    for k in range(int(n)):
                        actions += [j * size + k]

                actions += [size**2 + 1]

                for selected_action_idx in actions:

                    traj.append(((torch.clone(nodes[0]), torch.clone(edges[0]), torch.clone(masks[0])), selected_action_idx))

                    (nodes, edges, masks), done = self.update_state(nodes, edges, masks, torch.tensor([False]), torch.tensor([selected_action_idx]))

                    if done.item():
                        break

                traj.append(copy.deepcopy(traj[-1]))  # add padding state (will be ignored)
                traj[-1][0][0][:, -1] = 1
                self.precomputed.append(traj)

        return self.precomputed

    @torch.no_grad()
    def generate_graphs(self, num):

        if num == 0:
            return []

        self.base_model.eval(), self.stop_model.eval(), self.node_model.eval(), self.edge_model.eval()

        nodes = torch.zeros((num, self.start_size, self.node_features), device="cpu")
        edges = torch.zeros((num, self.start_size, self.start_size, self.edge_features), device="cpu")
        masks = torch.zeros((num, self.start_size), dtype=bool, device="cpu")

        nodes[:, 0, 0] = 1
        masks[:, 0] = 1

        done = torch.tensor([False] * num)

        for __ in range(self.max_len):

            embeddings, structure = get_embeddings(
                self.base_model,
                nodes[~done],
                edges[~done],
                masks[~done],
                device=self.device
            )

            action_probs = get_action_probs(
                *embeddings,
                *structure,
                self.stop_model,
                self.node_model,
                self.edge_model, 
                random_action_prob=0,
                apply_masks=True,
                max_nodes=self.max_nodes,
                masked_action_value=self.masked_action_value,
                action_prob_clip_bounds=self.action_prob_clip_bounds
            )

            assert action_probs.shape == (done.shape[0] - torch.sum(done), nodes.shape[1]**2 + 2)

            selected_action_idxs = torch.multinomial(torch.exp(action_probs), 1).reshape((-1,))
            (nodes, edges, masks), done = self.update_state(nodes, edges, masks, done, selected_action_idxs)

            if torch.all(done):
                break

        return [t for i, t in enumerate(zip(nodes, edges, masks)) if done[i]]

    @torch.no_grad()
    def get_log_unnormalised_ens(self, refl=False):

        self.base_model.eval(), self.stop_model.eval(), self.node_model.eval(), self.edge_model.eval()

        nodes = torch.zeros((1, self.start_size, self.node_features), device="cpu")
        edges = torch.zeros((1, self.start_size, self.start_size, self.edge_features), device="cpu")
        masks = torch.zeros((1, self.start_size), dtype=bool, device="cpu")

        nodes[:, 0, 0] = 1
        masks[:, 0] = 1
        edges[:, 0, 0, (0, 1)] = int(refl)  # need to update nodes as well for some node_history_bounds values

        embeddings, structure = get_embeddings(self.base_model, nodes, edges, masks, device=self.device)

        action_probs = get_action_probs(
            *embeddings,
            *structure,
            self.stop_model,
            self.node_model,
            self.edge_model, 
            random_action_prob=0,
            apply_masks=True,
            max_nodes=self.max_nodes,
            masked_action_value=self.masked_action_value,
            action_prob_clip_bounds=self.action_prob_clip_bounds
        )

        return action_probs[0, (0, -2, -1)]
