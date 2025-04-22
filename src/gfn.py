# TODO: backward policies to add
#  - optimal backward
#  - pessimistic gfns
#  - meta learnt
#  - increasing prob for actions under used in pf with some temp compared to optimal

from math import log
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from util import huber


@torch.no_grad
def get_num_previous_acts(state):  # (incorrect for starting state)
    nodes, edges, mask = state
    if nodes[0, -1].item() == 1:
        return 1
    edges = edges[mask][:, mask, 0]
    has_disconnected = (torch.sum(edges[:, -1]) == 0 and torch.sum(edges[-1, :]) == 0).item()
    return torch.sum(edges, dim=(0, 1)) + has_disconnected

def get_embeddings(base_model, nodes, edges, masks, device="cuda"):

    # extract only unique inputs
    combined = torch.concat((nodes.reshape(nodes.shape[0], 1, -1, nodes.shape[2]), edges), dim=1)
    output, inverse_indices = torch.unique(combined, return_inverse=True, dim=0)
    unique_nodes, unique_edges = torch.split(output, (1, edges.shape[1]), dim=1)
    unique_nodes = unique_nodes.reshape((unique_nodes.shape[0], -1, nodes.shape[2]))
    indices = torch.scatter_reduce(torch.full(size=(output.shape[0],), fill_value=inverse_indices.shape[0], device="cpu"), index=inverse_indices,
                                    src=torch.arange(inverse_indices.shape[0], device="cpu"), dim=0, reduce="amin")
    unique_masks = masks[indices]

    # get embeddings from the transformer
    node_embeddings, edge_embeddings = base_model(unique_nodes.to(device), unique_edges.to(device), mask=unique_masks.to(device))

    # add node embeddings to edges
    node_grid_a = node_embeddings.reshape((node_embeddings.shape[0], 1, *node_embeddings.shape[1:])).expand((node_embeddings.shape[0], node_embeddings.shape[1], *node_embeddings.shape[1:]))
    node_grid_b = node_embeddings.reshape((*node_embeddings.shape[:2], 1, node_embeddings.shape[2])).expand((*node_embeddings.shape[:2], node_embeddings.shape[1], node_embeddings.shape[2]))
    edge_embeddings = torch.concat((edge_embeddings, node_grid_a, node_grid_b), dim=3)

    node_embeddings[~unique_masks] = 0
    mean_embedding = torch.sum(node_embeddings, dim=1)  # use sum instead of mean because it makes more sense for this specific task

    return (node_embeddings, edge_embeddings, mean_embedding), (inverse_indices, unique_masks, unique_edges[:, :, :, 0] == 1)

def get_action_probs(
        node_embeddings, edge_embeddings, global_embedding,
        inverse_indices, unique_masks, unique_edges,
        stop_model, node_model, edge_model,
        random_action_prob=0, apply_masks=True, max_nodes=8,
        masked_action_value=-80, action_prob_clip_bounds=(-75, 75)
    ):

    # get action predictions from the smaller models
    stop_pred = stop_model(global_embedding).to("cpu")
    node_pred = node_model(global_embedding).to("cpu")
    edge_pred = edge_model(edge_embeddings.reshape((-1, edge_embeddings.shape[3]))).reshape((node_embeddings.shape[0], -1)).to("cpu")

    clipped_stop_pred = torch.clamp(stop_pred, min=action_prob_clip_bounds[0], max=action_prob_clip_bounds[1])
    clipped_node_pred = torch.clamp(node_pred, min=action_prob_clip_bounds[0], max=action_prob_clip_bounds[1])
    clipped_edge_pred = torch.clamp(edge_pred, min=action_prob_clip_bounds[0], max=action_prob_clip_bounds[1])

    rand_acts = random.choices(list(range(node_embeddings.shape[0])), k=round(node_embeddings.shape[0] * random_action_prob))
    clipped_stop_pred[rand_acts] = clipped_node_pred[rand_acts] = clipped_edge_pred[rand_acts] = 1

    if apply_masks:

        repeated_mask = unique_masks.reshape((*unique_masks.shape, 1)).expand(-1, -1, unique_masks.shape[1])
        edge_action_mask = torch.logical_and(torch.transpose(repeated_mask, 1, 2), repeated_mask)  # out-of-domain actions
        edge_action_mask[unique_edges] = False  # repeated actions
        edge_action_mask = edge_action_mask.reshape((node_embeddings.shape[0], -1))
        clipped_edge_pred[~edge_action_mask] = masked_action_value

        node_action_mask = torch.sum(unique_masks, dim=1) < max_nodes
        clipped_node_pred[~node_action_mask] = masked_action_value

    unique_action_probs = torch.concat((clipped_edge_pred, clipped_node_pred, clipped_stop_pred), dim=1)

    corr_unique_action_probs = unique_action_probs - torch.max(unique_action_probs)  # to avoid overflow
    norm_unique_action_probs = corr_unique_action_probs - torch.logsumexp(corr_unique_action_probs, dim=1).reshape((-1, 1))  # log softmax

    action_probs = norm_unique_action_probs[inverse_indices]

    return action_probs

def process_trajs(get_loss_fn):

    def get_loss_fn_from_trajs(jagged_trajs, log_rewards, base_model, log_z_model, *model_heads, device="cuda", **kwargs):

        for m in [base_model] + model_heads:
            m.train()

        # flatten inputs from jagged list
        traj_lens = torch.tensor([len(t) for t in jagged_trajs], device=device)
        trajs = [action for traj in jagged_trajs for action in traj]

        # pad inputs to the same length (this is quite memory intensive)
        nodes = nn.utils.rnn.pad_sequence([n for (n, _e, _m), _a in trajs], batch_first=True)
        edges = torch.stack([F.pad(e, (0, 0, 0, nodes.shape[1] - e.shape[1], 0, nodes.shape[1] - e.shape[0]), "constant", 0) for (_n, e, _m), _a in trajs])
        masks = torch.stack([F.pad(m, (0, nodes.shape[1] - m.shape[0]), "constant", 0) for (_n, _e, m), _a in trajs])

        pre_padding_lens = [n.shape[0] for (n, _e, _m), _a in trajs]
        post_padding_len = nodes.shape[1]
        actions = adjust_action_idxs(torch.tensor([a for _s, a in trajs]), pre_padding_lens, post_padding_len)
        
        embeddings = get_embeddings(base_model, nodes, edges, masks, device=device)
        log_z = log_z_model(torch.tensor([[1.]], device=device))

        return get_loss_fn(jagged_trajs, traj_lens, *embeddings, actions, log_z, log_rewards, *model_heads, device=device, **kwargs)

    return get_loss_fn_from_trajs

@process_trajs
def get_tb_loss_uniform(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        stop_model, node_model, edge_model,
        device="cuda"
    ):

    action_probs = get_action_probs(*raw_embeddings, *embedding_structure, stop_model, node_model, edge_model, random_action_prob=0, apply_masks=True)

    log_p_f = action_probs[list(range(len(action_probs))), actions]
    log_p_f[torch.cumsum(traj_lens, 0) - 1] = 0  # don't count padding states (kind of wasteful to compute these)

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = torch.tensor([-sum([log(get_num_previous_acts(s)) for s, _a in t[1:]]) for t in jagged_trajs], device=device)

    log_rewards = log_rewards.to(device)

    traj_diffs = (log_z + traj_log_p_f) - (log_rewards + traj_log_p_b)  # log_z gets broadcast into a vector here
    loss = huber(traj_diffs).mean()

    connected_prop = mean_num_nodes = 0
    for traj in jagged_trajs:
        nodes, edges, _mask = traj[-2][0]
        num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
        num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))
        connected_prop += (num_edges == num_nodes**2) / len(jagged_trajs)
        mean_num_nodes += num_nodes / len(jagged_trajs)

    return loss, {"log_z": log_z.item(), "mean_log_reward": torch.mean(log_rewards).item(), "connected_prop": connected_prop.item(),
                  "mean_num_nodes": mean_num_nodes.item()}

def adjust_action_idxs(action_idxs, pre_padding_lens, post_padding_len):
    for i in range(len(action_idxs)):
        preceeding_rows = action_idxs[i] // pre_padding_lens[i]
        action_idxs[i] += preceeding_rows * (post_padding_len - pre_padding_lens[i])
        if preceeding_rows == pre_padding_lens[i]:
            action_idxs[i] += (post_padding_len - pre_padding_lens[i]) * post_padding_len
    return action_idxs

@process_trajs
def get_tb_loss_add_node_mult(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        stop_model, node_model, edge_model,
        n=1, device="cuda"
    ):

    action_probs = get_action_probs(*raw_embeddings, *embedding_structure, stop_model, node_model, edge_model, random_action_prob=0, apply_masks=True)

    log_p_f = action_probs[list(range(len(action_probs))), actions]
    log_p_f[torch.cumsum(traj_lens, 0) - 1] = 0  # don't count padding states (kind of wasteful to compute these)

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = torch.tensor([-sum([log(get_num_previous_acts(s)) for s, _a in t[1:]]) for t in jagged_trajs], device=device)

    traj_log_p_b = torch.zeros((len(jagged_trajs),), device=device)
    for i, t in enumerate(jagged_trajs):
        for s, a in t[1:]:
            num = get_num_previous_acts(s)
            has_disconnected = (torch.sum(s[1][:, -1]) == 0 and torch.sum(s[1][-1, :]) == 0).item()
            if has_disconnected:
                if a == len(s[2])**2:
                    traj_log_p_b[i] += log(n/(num+n-1))
                else:
                    traj_log_p_b[i] += log(1/(num+n-1))
            else:
                traj_log_p_b[i] += log(1/num)

    log_rewards = log_rewards.to(device)

    traj_diffs = (log_z + traj_log_p_f) - (log_rewards + traj_log_p_b)  # log_z gets broadcast into a vector here
    loss = huber(traj_diffs).mean()

    connected_prop = mean_num_nodes = 0
    for traj in jagged_trajs:
        nodes, edges, _mask = traj[-2][0]
        num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
        num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))
        connected_prop += (num_edges == num_nodes**2) / len(jagged_trajs)
        mean_num_nodes += num_nodes / len(jagged_trajs)

    return loss, {"log_z": log_z.item(), "mean_log_reward": torch.mean(log_rewards).item(), "connected_prop": connected_prop.item(),
                  "mean_num_nodes": mean_num_nodes.item()}


@process_trajs
def get_tb_loss_tlm(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        fwd_stop_model, fwd_node_model, fwd_edge_model,
        bck_stop_model, bck_node_model, bck_edge_model,
        device="cuda"
    ):

    fwd_action_probs = get_action_probs(*raw_embeddings, *embedding_structure, fwd_stop_model, fwd_node_model, fwd_edge_model, random_action_prob=0, apply_masks=True)
    bck_action_probs = get_action_probs(*raw_embeddings, *embedding_structure, bck_stop_model, bck_node_model, bck_edge_model, random_action_prob=0, apply_masks=False)

    log_p_f = fwd_action_probs[list(range(len(fwd_action_probs))), actions]
    log_p_b = bck_action_probs[list(range(len(bck_action_probs))), torch.roll(actions, 1, 0)]  # prob of previous action
    log_p_b = torch.roll(log_p_b, -1, 0)  # we could save a roll here but it would be confusing

    final_graph_idxs = torch.cumsum(traj_lens, 0) - 1

    # don't count padding states (kind of wasteful to compute these)
    log_p_f[final_graph_idxs] = 0
    log_p_b[final_graph_idxs] = 0

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = scatter(log_p_b.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

    back_loss = -traj_log_p_b.mean()
    traj_log_p_b = traj_log_p_b.detach()

    log_rewards = log_rewards.to(device)

    traj_diffs = (log_z + traj_log_p_f) - (log_rewards + traj_log_p_b)  # log_z gets broadcast into a vector here
    tb_loss = huber(traj_diffs).mean()
    
    loss = tb_loss + back_loss

    connected_prop = mean_num_nodes = 0
    for traj in jagged_trajs:
        nodes, edges, _mask = traj[-2][0]
        num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
        num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))
        connected_prop += (num_edges == num_nodes**2) / len(jagged_trajs)
        mean_num_nodes += num_nodes / len(jagged_trajs)

    return loss, {"log_z": log_z.item(), "mean_log_reward": torch.mean(log_rewards).item(), "connected_prop": connected_prop.item(),
                  "mean_num_nodes": mean_num_nodes.item(), "tb_loss": tb_loss.detach(), "back_loss": back_loss.detach()}

@process_trajs
def get_tb_loss_smooth_tlm(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        fwd_stop_model, fwd_node_model, fwd_edge_model,
        bck_stop_model, bck_node_model, bck_edge_model,
        a=0.5,  # cosine annealing curve from 1 to 0 over time
        device="cuda"
    ):

    fwd_action_probs = get_action_probs(*raw_embeddings, *embedding_structure, fwd_stop_model, fwd_node_model, fwd_edge_model, random_action_prob=0, apply_masks=True)
    bck_action_probs = get_action_probs(*raw_embeddings, *embedding_structure, bck_stop_model, bck_node_model, bck_edge_model, random_action_prob=0, apply_masks=False)

    log_p_f = fwd_action_probs[list(range(len(fwd_action_probs))), actions]
    log_p_b = bck_action_probs[list(range(len(bck_action_probs))), torch.roll(actions, 1, 0)]  # prob of previous action
    log_p_b = torch.roll(log_p_b, -1, 0)  # we could save a roll here but it would be confusing

    final_graph_idxs = torch.cumsum(traj_lens, 0) - 1

    # don't count padding states (kind of wasteful to compute these)
    log_p_f[final_graph_idxs] = 0
    log_p_b[final_graph_idxs] = 0

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b_tlm = scatter(log_p_b.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

    back_loss = -traj_log_p_b_tlm.mean()
    traj_log_p_b_tlm = traj_log_p_b_tlm.detach()

    traj_unfiorm_log_p_b = torch.tensor([-sum([log(get_num_previous_acts(s)) for s, _a in t[1:]]) for t in jagged_trajs], device=device)
    traj_log_p_b = (1-a) * traj_log_p_b_tlm + (a) * traj_unfiorm_log_p_b

    log_rewards = log_rewards.to(device)

    traj_diffs = (log_z + traj_log_p_f) - (log_rewards + traj_log_p_b)  # log_z gets broadcast into a vector here
    tb_loss = huber(traj_diffs).mean()
    
    loss = tb_loss + back_loss

    connected_prop = mean_num_nodes = 0
    for traj in jagged_trajs:
        nodes, edges, _mask = traj[-2][0]
        num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
        num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))
        connected_prop += (num_edges == num_nodes**2) / len(jagged_trajs)
        mean_num_nodes += num_nodes / len(jagged_trajs)

    return loss, {"log_z": log_z.item(), "mean_log_reward": torch.mean(log_rewards).item(), "connected_prop": connected_prop.item(),
                  "mean_num_nodes": mean_num_nodes.item(), "tb_loss": tb_loss.detach(), "back_loss": back_loss.detach()}

@process_trajs
def get_tb_loss_biased_tlm(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        fwd_stop_model, fwd_node_model, fwd_edge_model,
        bck_stop_model, bck_node_model, bck_edge_model,
        multiplier=5, ns=[3, 4, 5],  # bias tlm backward policy towards fully connected graph with nodes counts in ns
        device="cuda"
    ):

    fwd_action_probs = get_action_probs(*raw_embeddings, *embedding_structure, fwd_stop_model, fwd_node_model, fwd_edge_model, random_action_prob=0, apply_masks=True)
    bck_action_probs = get_action_probs(*raw_embeddings, *embedding_structure, bck_stop_model, bck_node_model, bck_edge_model, random_action_prob=0, apply_masks=False)

    log_p_f = fwd_action_probs[list(range(len(fwd_action_probs))), actions]
    log_p_b = bck_action_probs[list(range(len(bck_action_probs))), torch.roll(actions, 1, 0)]  # prob of previous action
    log_p_b = torch.roll(log_p_b, -1, 0)  # we could save a roll here but it would be confusing

    final_graph_idxs = torch.cumsum(traj_lens, 0) - 1

    # don't count padding states (kind of wasteful to compute these)
    log_p_f[final_graph_idxs] = 0
    log_p_b[final_graph_idxs] = 0

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = scatter(log_p_b.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

    traj_log_p_b *= 1 + (multiplier - 1) * torch.tensor([int(is_n_connected(*t[-2][0], ns=ns)) for t in jagged_trajs])

    back_loss = -traj_log_p_b.mean()
    traj_log_p_b = traj_log_p_b.detach()

    log_rewards = log_rewards.to(device)

    traj_diffs = (log_z + traj_log_p_f) - (log_rewards + traj_log_p_b)  # log_z gets broadcast into a vector here
    tb_loss = huber(traj_diffs).mean()
    
    loss = tb_loss + back_loss

    connected_prop = mean_num_nodes = 0
    for traj in jagged_trajs:
        nodes, edges, _mask = traj[-2][0]
        num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
        num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))
        connected_prop += (num_edges == num_nodes**2) / len(jagged_trajs)
        mean_num_nodes += num_nodes / len(jagged_trajs)

    return loss, {"log_z": log_z.item(), "mean_log_reward": torch.mean(log_rewards).item(), "connected_prop": connected_prop.item(),
                  "mean_num_nodes": mean_num_nodes.item(), "tb_loss": tb_loss.detach(), "back_loss": back_loss.detach()}

@torch.no_grad
def is_n_connected(nodes, edges, mask, ns=[3]):
    num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=1)
    num_edges = torch.sum(edges[:, :, 0], dim=(1, 2))
    return num_nodes in ns and num_edges == num_nodes**2

@process_trajs
def get_tb_loss_free(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        fwd_stop_model, fwd_node_model, fwd_edge_model,
        bck_stop_model, bck_node_model, bck_edge_model,
        device="cuda"
    ):

    fwd_action_probs = get_action_probs(*raw_embeddings, *embedding_structure, fwd_stop_model, fwd_node_model, fwd_edge_model, random_action_prob=0, apply_masks=True)
    bck_action_probs = get_action_probs(*raw_embeddings, *embedding_structure, bck_stop_model, bck_node_model, bck_edge_model, random_action_prob=0, apply_masks=False)


    log_p_f = fwd_action_probs[list(range(len(fwd_action_probs))), actions]
    log_p_b = bck_action_probs[list(range(len(bck_action_probs))), torch.roll(actions, 1, 0)]  # prob of previous action
    log_p_b = torch.roll(log_p_b, -1, 0)  # we could save a roll here but it would be confusing

    final_graph_idxs = torch.cumsum(traj_lens, 0) - 1

    # don't count padding states (kind of wasteful to compute these)
    log_p_f[final_graph_idxs] = 0
    log_p_b[final_graph_idxs] = 0

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = scatter(log_p_b.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

    log_rewards = log_rewards.to(device)

    traj_diffs = (log_z + traj_log_p_f) - (log_rewards + traj_log_p_b)  # log_z gets broadcast into a vector here
    tb_loss = huber(traj_diffs).mean()
    
    loss = tb_loss

    connected_prop = mean_num_nodes = 0
    for traj in jagged_trajs:
        nodes, edges, _mask = traj[-2][0]
        num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
        num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))
        connected_prop += (num_edges == num_nodes**2) / len(jagged_trajs)
        mean_num_nodes += num_nodes / len(jagged_trajs)

    return loss, {"log_z": log_z.item(), "mean_log_reward": torch.mean(log_rewards).item(), "connected_prop": connected_prop.item(),
                  "mean_num_nodes": mean_num_nodes.item(), "tb_loss": tb_loss.detach()}

@process_trajs
def get_tb_loss_maxent(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        fwd_stop_model, fwd_node_model, fwd_edge_model,
        bck_stop_model, bck_node_model, bck_edge_model,
        n_model,
        device="cuda"
    ):

    fwd_action_probs = get_action_probs(*raw_embeddings, *embedding_structure, fwd_stop_model, fwd_node_model, fwd_edge_model, random_action_prob=0, apply_masks=True)
    bck_action_probs = get_action_probs(*raw_embeddings, *embedding_structure, bck_stop_model, bck_node_model, bck_edge_model, random_action_prob=0, apply_masks=False)

    log_p_f = fwd_action_probs[list(range(len(fwd_action_probs))), actions]
    log_p_b = bck_action_probs[list(range(len(bck_action_probs))), torch.roll(actions, 1, 0)]  # prob of previous action
    log_p_b = torch.roll(log_p_b, -1, 0)  # we could save a roll here but it would be confusing

    final_graph_idxs = torch.cumsum(traj_lens, 0) - 1
    first_graph_idxs = torch.roll(final_graph_idxs, 1, dims=0) + 1
    first_graph_idxs[0] = 0

    # don't count padding states (kind of wasteful to compute these)
    log_p_f[final_graph_idxs] = 0
    log_p_b[final_graph_idxs] = 0

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = scatter(log_p_b.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

    traj_pred_l = n_model(raw_embeddings[2][final_graph_idxs - 1]).to("cpu")
    traj_pred_l[first_graph_idxs >= final_graph_idxs - 1] = 0

    n_loss = huber(traj_log_p_b + traj_pred_l).mean()
    traj_log_p_b = traj_log_p_b.detach()

    log_rewards = log_rewards.to(device)

    traj_diffs = (log_z + traj_log_p_f) - (log_rewards + traj_log_p_b)  # log_z gets broadcast into a vector here
    tb_loss = huber(traj_diffs).mean()

    loss = tb_loss + n_loss

    connected_prop = mean_num_nodes = 0
    for traj in jagged_trajs:
        nodes, edges, _mask = traj[-2][0]
        num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
        num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))
        connected_prop += (num_edges == num_nodes**2) / len(jagged_trajs)
        mean_num_nodes += num_nodes / len(jagged_trajs)

    return loss, {"log_z": log_z.item(), "mean_log_reward": torch.mean(log_rewards).item(), "connected_prop": connected_prop.item(),
                  "mean_num_nodes": mean_num_nodes.item(), "tb_loss": tb_loss.detach(), "n_loss": n_loss.detach()}

@process_trajs
def get_tb_loss_rand_const(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        stop_model, node_model, edge_model,
        mean=0.2, std=0.125, eps=0.01, precision=1_000,
        device="cuda"
    ):

    action_probs = get_action_probs(*raw_embeddings, *embedding_structure, stop_model, node_model, edge_model, random_action_prob=0, apply_masks=True)

    log_p_f = action_probs[list(range(len(action_probs))), actions]
    log_p_b = torch.tensor([hash(s) % precision for traj in jagged_trajs for s, _a in traj])
    log_p_b = torch.log(torch.clamp((log_p_b / precision - 0.5) * (std * 2) + mean, min=eps, max=1))
    log_p_b = torch.roll(log_p_b, -1, 0)

    final_graph_idxs = torch.cumsum(traj_lens, 0) - 1

    # don't count padding states (kind of wasteful to compute these)
    log_p_f[final_graph_idxs] = 0
    log_p_b[final_graph_idxs] = 0

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = scatter(log_p_b.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

    traj_log_p_b = traj_log_p_b.detach()  # probably not necessary
    log_rewards = log_rewards.to(device)

    traj_diffs = (log_z + traj_log_p_f) - (log_rewards + traj_log_p_b)  # log_z gets broadcast into a vector here
    loss = huber(traj_diffs).mean()

    connected_prop = mean_num_nodes = 0
    for traj in jagged_trajs:
        nodes, edges, _mask = traj[-2][0]
        num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
        num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))
        connected_prop += (num_edges == num_nodes**2) / len(jagged_trajs)
        mean_num_nodes += num_nodes / len(jagged_trajs)

    return loss, {"log_z": log_z.item(), "mean_log_reward": torch.mean(log_rewards).item(), "connected_prop": connected_prop.item(),
                  "mean_num_nodes": mean_num_nodes.item()}

@process_trajs
def get_tb_loss_rand_var(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        stop_model, node_model, edge_model,
        mean=0.2, std=0.125, eps=0.01,
        device="cuda"
    ):

    action_probs = get_action_probs(*raw_embeddings, *embedding_structure, stop_model, node_model, edge_model, random_action_prob=0, apply_masks=True)

    log_p_f = action_probs[list(range(len(action_probs))), actions]
    log_p_b = torch.normal(mean=mean, std=std, size=(raw_embeddings[2].shape[0],))  # could compute traj_log_p_b directly if we don't want to try other distributions
    log_p_b = torch.log(torch.clamp(log_p_b, min=eps, max=1))

    final_graph_idxs = torch.cumsum(traj_lens, 0) - 1

    # don't count padding states (kind of wasteful to compute these)
    log_p_f[final_graph_idxs] = 0
    log_p_b[final_graph_idxs] = 0

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = scatter(log_p_b.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

    traj_log_p_b = traj_log_p_b.detach()  # probably not necessary
    log_rewards = log_rewards.to(device)

    traj_diffs = (log_z + traj_log_p_f) - (log_rewards + traj_log_p_b)  # log_z gets broadcast into a vector here
    loss = huber(traj_diffs).mean()

    connected_prop = mean_num_nodes = 0
    for traj in jagged_trajs:
        nodes, edges, _mask = traj[-2][0]
        num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
        num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))
        connected_prop += (num_edges == num_nodes**2) / len(jagged_trajs)
        mean_num_nodes += num_nodes / len(jagged_trajs)

    return loss, {"log_z": log_z.item(), "mean_log_reward": torch.mean(log_rewards).item(), "connected_prop": connected_prop.item(),
                  "mean_num_nodes": mean_num_nodes.item()}

@process_trajs
def get_tb_loss_const(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        stop_model, node_model, edge_model,
        value=0.2,
        device="cuda"
    ):

    action_probs = get_action_probs(*raw_embeddings, *embedding_structure, stop_model, node_model, edge_model, random_action_prob=0, apply_masks=True)

    log_p_f = action_probs[list(range(len(action_probs))), actions]
    log_p_b = torch.full((raw_embeddings[2].shape[0],), log(value))

    final_graph_idxs = torch.cumsum(traj_lens, 0) - 1

    # don't count padding states (kind of wasteful to compute these)
    log_p_f[final_graph_idxs] = 0
    log_p_b[final_graph_idxs] = 0

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = torch.full((traj_lens.shape[0],), log(value)) * traj_lens

    traj_log_p_b = traj_log_p_b.detach()  # probably not necessary
    log_rewards = log_rewards.to(device)

    traj_diffs = (log_z + traj_log_p_f) - (log_rewards + traj_log_p_b)  # log_z gets broadcast into a vector here
    loss = huber(traj_diffs).mean()

    connected_prop = mean_num_nodes = 0
    for traj in jagged_trajs:
        nodes, edges, _mask = traj[-2][0]
        num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
        num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))
        connected_prop += (num_edges == num_nodes**2) / len(jagged_trajs)
        mean_num_nodes += num_nodes / len(jagged_trajs)

    return loss, {"log_z": log_z.item(), "mean_log_reward": torch.mean(log_rewards).item(), "connected_prop": connected_prop.item(),
                  "mean_num_nodes": mean_num_nodes.item()}
