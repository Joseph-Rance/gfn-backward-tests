import math
import copy
import numpy as np
import networkx as nx
import torch
from torch.utils.data import IterableDataset

from gfn import get_embeddings, get_action_probs


def get_uniform_counting_log_reward(nodes, _edges, **kwargs):  # this is like counting from that other paper but reversed for efficiency
    num_nodes = torch.sum(torch.sum(nodes, dim=2) > 0, dim=1)
    return torch.clamp(- math.log(2) * num_nodes ** 2, max=1_000, min=-1_000)

def get_smoothed_overfit_log_reward(nodes, edges, reward_arg=0.8, alpha=10, **kwargs):
    num_nodes = torch.sum(torch.sum(nodes, dim=2) > 0, dim=1)
    num_edges = torch.sum(edges[:, :, :, 0], dim=(1, 2))
    #return torch.logical_and(num_nodes == 2, num_edges == 0).long()  # (for testing)
    fully_connectedness = (num_edges - num_nodes**2) ** 2
    return torch.clamp(math.log(reward_arg) * num_nodes - alpha * fully_connectedness, min=-1_000)

def get_cliques_log_reward(nodes, edges, reward_arg=3, m=10, eta=0.00000000001, **kwargs):  # reward is ReLU( m * # nodes in exactly 1 n-clique - # edges )
    num_nodes = torch.sum(torch.sum(nodes, dim=2) > 0, dim=1)
    num_edges = torch.sum(edges[:, :, :, 0], dim=(1, 2))
    log_rewards = []
    for i in range(len(nodes)):
        adj_matrix = edges[i, :num_nodes[i], :num_nodes[i], 0]
        g = nx.from_numpy_array(adj_matrix.cpu().numpy(), edge_attr=None)  # does not include create_using=nx.DiGraph, so we convert to an undirected graph
        n_cliques = [c for c in nx.algorithms.clique.find_cliques(g) if len(c) == reward_arg]
        n_cliques_per_node = np.bincount(sum(n_cliques, []), minlength=num_nodes[i])
        reward = (np.sum(n_cliques_per_node == 1) * m - num_edges[i]) * 1e12 / math.exp(0.6 * num_nodes[i] ** 2 + 1.2 * num_nodes[i])
        reward = max(reward, eta)
        log_rewards.append(math.log(reward))
    return torch.tensor(log_rewards)

def get_reward_fn_generator(reward_fn, reward_arg=0.8, alpha_start=1_000_000, alpha_change=1):
    alpha = alpha_start
    while True:
        yield lambda *args, **kwargs: reward_fn(*args, reward_arg=reward_arg, alpha=alpha, **kwargs)
        alpha *= alpha_change  # it is probably a bad idea to actually have alpha_change != 1 as log(z) would keep changing
        alpha = min(alpha, 1_000_000)


#  nodes in exactly 1 3-clique  0                   1            2            3                   4             5             6             7
uniform_true_dist = np.array([[[0, 0],             [0, 0],      [0, 0],      [0, 0],             [0, 0],       [0, 0],       [0, 0],       [0, 0]],   # graphs with 0 nodes
                              [[0.07143, 0.07143], [0, 0],      [0, 0],      [0, 0],             [0, 0],       [0, 0],       [0, 0],       [0, 0]],   #             1
                              [[0.13393, 0.00893], [0, 0],      [0, 0],      [0, 0],             [0, 0],       [0, 0],       [0, 0],       [0, 0]],   #             2
                              [[0.08259, 0],       [0, 0],      [0, 0],      [0.05999, 0.00028], [0, 0],       [0, 0],       [0, 0],       [0, 0]],   #             3
                              [[0.05434, 0],       [0, 0],      [0.05085, 0],[0.03767, 0],       [0, 0],       [0, 0],       [0, 0],       [0, 0]],   #             4
                              [[0.06011, 0],       [0.02682, 0],[0.03973, 0],[0.01471, 0],       [0.00149, 0], [0, 0],       [0, 0],       [0, 0]],   #             5
                              [[0.08649, 0],       [0.03080, 0],[0.01941, 0],[0.00517, 0],       [0.00065, 0], [0, 0],       [0.00034, 0], [0, 0]],   #             6
                              [[0.10979, 0],       [0.02422, 0],[0.00673, 0],[0.00176, 0],       [0.00021, 0], [0.00010, 0], [0.00005, 0], [0, 0]]])  #             7

overfit_true_dist = np.array([[[0, 0],       [0, 0], [0, 0], [0, 0],       [0, 0], [0, 0], [0, 0], [0, 0]],
                              [[0, 0.25307], [0, 0], [0, 0], [0, 0],       [0, 0], [0, 0], [0, 0], [0, 0]],
                              [[0, 0.20246], [0, 0], [0, 0], [0, 0],       [0, 0], [0, 0], [0, 0], [0, 0]],
                              [[0, 0],       [0, 0], [0, 0], [0, 0.16197], [0, 0], [0, 0], [0, 0], [0, 0]],
                              [[0, 0.12957], [0, 0], [0, 0], [0, 0],       [0, 0], [0, 0], [0, 0], [0, 0]],
                              [[0, 0.10366], [0, 0], [0, 0], [0, 0],       [0, 0], [0, 0], [0, 0], [0, 0]],
                              [[0, 0.08293], [0, 0], [0, 0], [0, 0],       [0, 0], [0, 0], [0, 0], [0, 0]],
                              [[0, 0.06634], [0, 0], [0, 0], [0, 0],       [0, 0], [0, 0], [0, 0], [0, 0]]])

cliques_true_dist = np.array([[[0, 0], [0, 0],       [0, 0],       [0, 0],             [0, 0],       [0, 0],       [0, 0],       [0, 0]],
                              [[0, 0], [0, 0],       [0, 0],       [0, 0],             [0, 0],       [0, 0],       [0, 0],       [0, 0]],
                              [[0, 0], [0, 0],       [0, 0],       [0, 0],             [0, 0],       [0, 0],       [0, 0],       [0, 0]],
                              [[0, 0], [0, 0],       [0, 0],       [0.50778, 0.00236], [0, 0],       [0, 0],       [0, 0],       [0, 0]],
                              [[0, 0], [0, 0],       [0.13825, 0], [0.17818, 0],       [0, 0],       [0, 0],       [0, 0],       [0, 0]],
                              [[0, 0], [0.00677, 0], [0.06695, 0], [0.04504, 0],       [0.00639, 0], [0, 0],       [0, 0],       [0, 0]],
                              [[0, 0], [0.00028, 0], [0.02196, 0], [0.01181, 0],       [0.00222, 0], [0, 0],       [0.00187, 0], [0, 0]],
                              [[0, 0], [0, 0],       [0.00537, 0], [0.00344, 0],       [0.00067, 0], [0.00040, 0], [0.00025, 0], [0, 0]]])


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
        adjust_random=None,
        max_len=500,
        max_nodes=10,
        batch_size=128,
        num_precomputed=0,
        edges_first=False,
        max_precomputed_len=4,
        reward_arg=0.8,
        undirected=False,
        node_history_bounds=(0, 1),  # inclusive
        edge_history_bounds=(0, 1),  # inclusive
        masked_action_value=-80,
        action_prob_clip_bounds=(-75, 75),  # inclusive
        replay_buffer_length=1024,
        replay_buffer_growth=128,
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
        self.adjust_random = adjust_random
        self.max_len = max_len
        self.max_nodes = max_nodes
        self.batch_size = batch_size
        self.num_precomputed = num_precomputed  # precomputed only works with the overfitted reward
        self.edges_first = edges_first
        self.max_precomputed_len = max_precomputed_len
        self.reward_arg = reward_arg
        self.undirected = undirected  # this works terribly for some reason
        self.node_history_bounds = node_history_bounds
        self.edge_history_bounds = edge_history_bounds
        self.masked_action_value = masked_action_value
        self.action_prob_clip_bounds = action_prob_clip_bounds
        self.replay_buffer_length = replay_buffer_length
        self.replay_buffer_growth = replay_buffer_growth
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
            idxs = torch.randint(0, max_idx, (self.batch_size - self.num_precomputed,))

            trajs += self.replay_buffer_trajs[idxs].tolist()
            log_rewards = torch.concatenate((log_rewards, self.replay_buffer_log_rewards[idxs]))

        if self.num_precomputed > 0:
            precomputed_trajs, precomputed_log_rewards = self.get_precomputed()
            trajs += precomputed_trajs
            log_rewards = torch.concatenate((log_rewards, precomputed_log_rewards))

        return trajs, log_rewards

    def get_sampled(self, num=None, test=False):

        if not test:
            self.base_model.eval(), self.stop_model.eval(), self.node_model.eval(), self.edge_model.eval()

        if num is not None:
            sample_size = num
        else:
            num_to_gen = max(self.replay_buffer_growth, self.batch_size - self.replay_end if not self.replay_saturated else 0)
            sample_size = num_to_gen - self.num_precomputed

        trajs = [[] for __ in range(sample_size)]

        nodes = torch.zeros((sample_size, self.start_size, self.node_features), device="cpu")
        edges = torch.zeros((sample_size, self.start_size, self.start_size, self.edge_features), device="cpu")
        masks = torch.zeros((sample_size, self.start_size), dtype=bool, device="cpu")

        nodes[:, 0, 0] = 1
        masks[:, 0] = 1

        if self.undirected:
            edges[:, 0, 0, (0, 1)] = 1

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
                random_action_prob=self.random_action_prob * (1-test),
                adjust_random=self.adjust_random,
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
        #nodes[:, :, 0] += torch.sum(nodes, dim=2) > 0  # increment existing nodes
        #nodes = torch.clamp(nodes, min=self.node_history_bounds[0], max=self.node_history_bounds[1])  # clip node histories to control tree property

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

            if self.undirected:
                edges[idxs, next_node[idxs], next_node[idxs], 0] = edges[idxs, next_node[idxs], next_node[idxs], 1] = 1

        # update edges
        edges[:, :, :, 1] += (torch.sum(edges, dim=3) > 0).float()  # increment existing nodes
        edges[:, :, :, 1] = torch.clamp(edges[:, :, :, 1], min=self.edge_history_bounds[0], max=self.edge_history_bounds[1])  # clip edge histories to control tree property

        j = 0
        for i in range(sample_size):  # too lazy to vectorise
            if done[i]:
                continue
            if selected_action_idxs[j] < prev_len**2:
                edges[i, selected_action_idxs[j] // prev_len, selected_action_idxs[j] % prev_len, (0, 1)] = 1
                if self.undirected:
                    edges[i, selected_action_idxs[j] % prev_len, selected_action_idxs[j] // prev_len, (0, 1)] = 1
            j += 1

        # update done
        done[~done] = (selected_action_idxs == prev_len**2 + 1).to("cpu")

        return (nodes, edges, masks), done

    def get_precomputed(self, importance_sample=True, base_multiplier=1.15):

        trajs = []

        if importance_sample:
            p_base = 1 - (1-self.reward_arg / base_multiplier)
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

        log_rewards = math.log(self.reward_arg) * num_nodes

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

                if not self.edges_first:

                    size = self.start_size
                    for n_i in range(int(n)-1):
                        actions += [size**2]
                        if n_i+1 == size:
                            size *= self.expand_factor

                    for j in range(int(n)):
                        for k in range(int(n)):
                            actions += [j * size + k]

                    actions += [size**2 + 1]
                
                else:

                    size = self.start_size

                    actions += [0]

                    for n_i in range(int(n)-1):
                        actions += [size**2, (n_i + 1) * size + n_i + 1]
                        for i in range(n_i + 1):
                            actions += [(n_i + 1) * size + i, i * size + (n_i + 1)]
                        if n_i+1 == size:
                            size *= self.expand_factor

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
