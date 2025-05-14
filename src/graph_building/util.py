import math
import torch


@torch.no_grad()
def get_num_previous_acts(state):  # (incorrect for starting state)
    nodes, edges, mask = state
    if nodes[0, -1].item() == 1:
        return 1
    edges = edges[mask][:, mask, 0]  # is this necessary?
    num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=1)
    has_disconnected = (torch.sum(edges[:, num_nodes-1]) == 0 and torch.sum(edges[num_nodes-1, :]) == 0).item()
    return torch.sum(edges, dim=(0, 1)) + has_disconnected

def adjust_action_idxs(action_idxs, pre_padding_lens, post_padding_len):
    for i in range(len(action_idxs)):
        preceeding_rows = action_idxs[i] // pre_padding_lens[i]
        action_idxs[i] += preceeding_rows * (post_padding_len - pre_padding_lens[i])
        if preceeding_rows == pre_padding_lens[i]:
            action_idxs[i] += (post_padding_len - pre_padding_lens[i]) * post_padding_len
    return action_idxs

@torch.no_grad()
def is_n_connected(nodes, edges, _mask, ns=[3]):
    num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=1)
    num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))
    return num_nodes in ns and num_edges == num_nodes**2

@torch.no_grad()
def get_aligned_action_log_prob(nodes, edges, _mask, action_idx, reward_idx=1, reward_arg=0.8, correct_prob=0.99, incorrect_prob=0.01, max_nodes=10):
    # get actions following handmade policy (see notepad)
    # this is completely unnormalised for a backward policy unless we have found the aligned forward policy

    if reward_idx == 0:

        num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
        num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))

        if num_edges == 0:  # first add nodes  ...
            if action_idx == num_nodes ** 2: # add node
                return math.log((1 * correct_prob + (max_nodes-num_nodes) * incorrect_prob) / (max_nodes-num_nodes+1))
            else:  # add edge or stop -> share correct prob between remaining actions
                return math.log((1 * incorrect_prob + (max_nodes-num_nodes) * correct_prob) / (max_nodes-num_nodes+1))

        else:  # ... then add edges

            if action_idx == num_nodes ** 2: # add node
                return math.log(incorrect_prob)
            else:  # add edge or stop -> share correct prob between remaining actions
                return math.log(correct_prob / (1 + num_nodes ** 2 - num_edges))  # could be more precise here

    if reward_idx == 1:

        num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
        num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))

        if num_edges < num_nodes ** 2:
            aligned_action_idx = (num_edges // num_nodes) * nodes.shape[0] + (num_edges % num_nodes)
            return math.log(correct_prob) if action_idx == aligned_action_idx else math.log(incorrect_prob)
        elif action_idx == num_nodes ** 2: # if we are fully connected, add node with probability reward_arg
            return reward_arg * math.log(correct_prob) + (1 - reward_arg) * math.log(incorrect_prob)
        elif action_idx == num_nodes ** 2 + 1:  # if we are fully connected, stop with probability 1-reward_arg
            return (1 - reward_arg) * math.log(correct_prob) + reward_arg * math.log(incorrect_prob)
        else:
            return math.log(incorrect_prob)

    else:
        raise ValueError("aligned policy only implemented for reward index in {1, 2}")

def get_state_hash(v):  # v is (state, action, seed)
    obj = (tuple(tuple(s.reshape(-1).tolist()) for s in v[0]), *v[1:])
    #hashGen = hashlib.md5()
    #hashGen.update((abs(hash(obj))).to_bytes(10, 'little'))
    #return int(hashGen.hexdigest(), 16)
    return hash(obj)

def huber(x, beta=1, i_delta=4):
    ax = torch.abs(x)
    return torch.where(ax <= beta, 0.5 * x * x, beta * (ax - beta / 2)) * i_delta
