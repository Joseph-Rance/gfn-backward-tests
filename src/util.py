import math
import torch


@torch.no_grad()
def get_num_previous_acts(state):  # (incorrect for starting state)
    nodes, edges, mask = state
    if nodes[0, -1].item() == 1:
        return 1
    edges = edges[mask][:, mask, 0]
    has_disconnected = (torch.sum(edges[:, -1]) == 0 and torch.sum(edges[-1, :]) == 0).item()
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
    num_edges = torch.sum(edges[:, :, 0], dim=(1, 2))
    return num_nodes in ns and num_edges == num_nodes**2

@torch.no_grad()
def get_aligned_action_log_prob(nodes, edges, _mask, action_idx, b=0.8, correct_log_prob=math.log(0.9), incorrect_log_prob=math.log(0.02)):  # get actions following handmade policy (see notepad)

    num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
    num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))

    if num_edges < num_nodes ** 2:
        aligned_action_idx = (num_edges // num_nodes) * nodes.shape[0] + (num_edges % num_nodes)
        return correct_log_prob if action_idx == aligned_action_idx else incorrect_log_prob
    elif action_idx == num_nodes ** 2: # if we are fully connected, add node with probability b
        return b * correct_log_prob + (1 - b) * incorrect_log_prob
    elif action_idx == num_nodes ** 2 + 1:  # if we are fully connected, stop with probability 1-b
        return (1 - b) * correct_log_prob + b * incorrect_log_prob
    else:
        return incorrect_log_prob

def get_prob_change(nodes, edges, _mask, action_idx, pred_prob, b=0.8):  # add output pred_prob

    capacity = nodes.shape[0]
    num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
    num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))

    if action_idx == capacity ** 2 + 1:  # stop

        if num_edges == num_nodes**2:
            return (1-b) - pred_prob

        return -pred_prob  # 0 flow to stopping on incomplete graphs

    elif action_idx == capacity ** 2:  # node

        if num_edges == num_nodes**2:
            return b - pred_prob

        return min(0, b - pred_prob)  # don't assign more than b flow to bigger graphs
    else:

        assert action_idx < capacity ** 2

        if num_edges == num_nodes**2:
            return -pred_prob

        return 0  # ok to add an edge so long as there are options

def get_state_hash(v):  # v is (state, action, seed)
    obj = (tuple(tuple(s.reshape(-1).tolist()) for s in v[0]), *v[1:])
    #hashGen = hashlib.md5()
    #hashGen.update((abs(hash(obj))).to_bytes(10, 'little'))
    #return int(hashGen.hexdigest(), 16)
    return hash(obj)

def huber(x, beta=1, i_delta=4):
    ax = torch.abs(x)
    return torch.where(ax <= beta, 0.5 * x * x, beta * (ax - beta / 2)) * i_delta
