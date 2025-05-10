import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from util import get_num_previous_acts, adjust_action_idxs, is_n_connected, get_aligned_action_log_prob, get_prob_change, get_state_hash, huber


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
    mean_embedding = torch.mean(node_embeddings, dim=1)

    return (node_embeddings, edge_embeddings, mean_embedding), (inverse_indices, unique_masks, unique_edges[:, :, :, 0] == 1)

def get_action_probs(
        node_embeddings, edge_embeddings, global_embedding,
        inverse_indices, unique_masks, unique_edges,
        stop_model, node_model, edge_model,
        random_action_prob=0, adjust_random=None, apply_masks=True, max_nodes=8,
        masked_action_value=-80, action_prob_clip_bounds=(-75, 75)
    ):

    # get action predictions from the smaller models
    stop_pred = stop_model(global_embedding).to("cpu")
    node_pred = node_model(global_embedding).to("cpu")
    edge_pred = edge_model(edge_embeddings.reshape((-1, edge_embeddings.shape[3]))).reshape((node_embeddings.shape[0], -1)).to("cpu")

    clipped_stop_pred = torch.clamp(stop_pred, min=action_prob_clip_bounds[0], max=action_prob_clip_bounds[1])
    clipped_node_pred = torch.clamp(node_pred, min=action_prob_clip_bounds[0], max=action_prob_clip_bounds[1])
    clipped_edge_pred = torch.clamp(edge_pred, min=action_prob_clip_bounds[0], max=action_prob_clip_bounds[1])

    rand_acts = random.sample(list(range(node_embeddings.shape[0])), k=round(node_embeddings.shape[0] * random_action_prob))
    clipped_stop_pred[rand_acts] = clipped_node_pred[rand_acts] = clipped_edge_pred[rand_acts] = 1

    if apply_masks:

        repeated_mask = unique_masks.reshape((*unique_masks.shape, 1)).expand(-1, -1, unique_masks.shape[1])
        edge_action_mask = torch.logical_and(torch.transpose(repeated_mask, 1, 2), repeated_mask)  # out-of-domain actions
        edge_action_mask[unique_edges] = False  # repeated actions
        edge_action_mask = edge_action_mask.reshape((node_embeddings.shape[0], -1))
        node_action_mask = torch.sum(unique_masks, dim=1) < max_nodes
    
    else:

        edge_action_mask = torch.ones(clipped_edge_pred.shape, dtype=bool)
        node_action_mask = torch.ones(clipped_node_pred.shape, dtype=bool)

    clipped_edge_pred[~edge_action_mask] = masked_action_value
    clipped_node_pred[~node_action_mask] = masked_action_value

    if adjust_random:

        adjusted_clipped_edge_pred = torch.clone(clipped_edge_pred)
        adjusted_clipped_stop_pred = torch.clone(clipped_stop_pred)

        random_edge_action_mask = torch.zeros(clipped_edge_pred.shape, dtype=bool)
        random_edge_action_mask[rand_acts] = edge_action_mask[rand_acts]

        adjusted_clipped_edge_pred[random_edge_action_mask] = (torch.div(
            clipped_edge_pred.T,
            torch.sum(clipped_edge_pred * random_edge_action_mask, dim=1)
        ).T[random_edge_action_mask] * adjust_random).detach()

        adjusted_clipped_stop_pred[rand_acts] = (adjusted_clipped_stop_pred[rand_acts] / adjust_random).detach()

    else:

        adjusted_clipped_edge_pred = clipped_edge_pred
        adjusted_clipped_stop_pred = clipped_stop_pred

    unique_action_probs = torch.concat((adjusted_clipped_edge_pred, clipped_node_pred, adjusted_clipped_stop_pred), dim=1)

    corr_unique_action_probs = unique_action_probs - torch.max(unique_action_probs)  # to avoid overflow
    norm_unique_action_probs = corr_unique_action_probs - torch.logsumexp(corr_unique_action_probs, dim=1).reshape((-1, 1))  # log softmax

    action_probs = norm_unique_action_probs[inverse_indices]

    return action_probs

def process_trajs(get_loss_fn):

    def get_loss_fn_from_trajs(jagged_trajs, log_rewards, base_models, log_z_model, *model_heads, constant_log_z=None, device="cuda", **kwargs):

        for m in (*base_models, *model_heads):
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

        log_z = torch.tensor(constant_log_z) if constant_log_z is not None else log_z_model(torch.tensor([[1.]], device=device))

        embeddings = []
        for base_model in base_models:
            embeddings.append(get_embeddings(base_model, nodes, edges, masks, device=device))

        return get_loss_fn(jagged_trajs, traj_lens, *list(zip(*embeddings)), actions, log_z, log_rewards, *model_heads, device=device, **kwargs)

    return get_loss_fn_from_trajs

@process_trajs
def get_loss_to_uniform_backward(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, _log_z, _log_rewards,
        _stop_model, _node_model, _edge_model,
        bck_stop_model, bck_node_model, bck_edge_model,
        device="cuda"
    ):

    trajs = [action for traj in jagged_trajs for action in traj]
    uniform_p_b = torch.tensor([math.log(get_num_previous_acts(s)) for s, _a in trajs], device=device)

    bck_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], bck_stop_model, bck_node_model, bck_edge_model, random_action_prob=0, apply_masks=True)
    log_p_b = bck_action_probs[list(range(len(bck_action_probs))), torch.roll(actions, 1, 0)].to(device)  # prob of previous action

    first_graph_idxs = torch.cumsum(traj_lens, 0)

    # don't count padding states (kind of wasteful to compute these)
    uniform_p_b[first_graph_idxs] = 0
    log_p_b[first_graph_idxs] = 0

    diffs = uniform_p_b - log_p_b
    loss = (diffs * diffs).mean()

    return loss

@process_trajs
def get_tb_loss_uniform(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        stop_model, node_model, edge_model,
        device="cuda"
    ):

    action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], stop_model, node_model, edge_model, random_action_prob=0, apply_masks=True)

    log_p_f = action_probs[list(range(len(action_probs))), actions]
    log_p_f[torch.cumsum(traj_lens, 0) - 1] = 0  # don't count padding states (kind of wasteful to compute these)

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = torch.tensor([-sum([math.log(get_num_previous_acts(s)) for s, _a in t[1:]]) for t in jagged_trajs], device=device)

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
                  "mean_num_nodes": mean_num_nodes.item(), "tb_loss": loss.detach()}

@process_trajs
def get_tb_loss_action_mult(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        stop_model, node_model, edge_model,
        action_type=0, n=1, device="cuda"
    ):

    action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], stop_model, node_model, edge_model, random_action_prob=0, apply_masks=True)

    log_p_f = action_probs[list(range(len(action_probs))), actions]
    log_p_f[torch.cumsum(traj_lens, 0) - 1] = 0  # don't count padding states (kind of wasteful to compute these)

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = torch.zeros((len(jagged_trajs),), device=device)

    for i, t in enumerate(jagged_trajs):
        for (__, a), (s, __) in zip(t[:-1], t[1:]):
            num = get_num_previous_acts(s)

            if action_type == 0:  # edge

                num_edges = torch.sum(s[1][:, :, 0], dim=(0, 1))
                if a < len(s[2])**2:  # if action is add edge, assign more weight
                    traj_log_p_b[i] += math.log(n/(num+num_edges*(n-1)))
                else:  # otherwise, assign normal weight
                    traj_log_p_b[i] += math.log(1/(num+num_edges*(n-1)))  # we need to adjust for the added weight of a potential add edge

            elif action_type == 1:  # node

                if a == len(s[2])**2:  # if action is add node, assign more weight
                    traj_log_p_b[i] += math.log(n/(num+n-1))
                else:  # otherwise, assign normal weight
                    num_nodes = torch.sum(torch.sum(s[0], dim=1) > 0, dim=1)
                    has_disconnected = int((torch.sum(s[1][:, num_nodes-1]) == 0 and torch.sum(s[1][num_nodes-1, :]) == 0).item())
                    traj_log_p_b[i] += math.log(1/(num+has_disconnected*(n-1)))  # we need to adjust for the added weight of a potential add node

            else:  # stop

                assert action_type == 2
                if a == len(s[2])**2 + 1:  # if action is stop, assign more weight
                    traj_log_p_b[i] += math.log(n/(num+n-1))
                else:  # otherwise, assign normal weight (no need to adjust for a potential use of stop before a non-terminal state)
                    traj_log_p_b[i] += math.log(1/num)

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
                  "mean_num_nodes": mean_num_nodes.item(), "tb_loss": loss.detach()}

@process_trajs
def get_tb_loss_tlm(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        fwd_stop_model, fwd_node_model, fwd_edge_model,
        bck_stop_model, bck_node_model, bck_edge_model,
        device="cuda"
    ):

    fwd_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], fwd_stop_model, fwd_node_model, fwd_edge_model, random_action_prob=0, apply_masks=True)
    bck_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], bck_stop_model, bck_node_model, bck_edge_model, random_action_prob=0, apply_masks=False)

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
def get_tb_loss_soft_tlm(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        fwd_stop_model, fwd_node_model, fwd_edge_model,
        bck_stop_model, bck_node_model, bck_edge_model,
        a=0.5,
        device="cuda"
    ):

    fwd_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], fwd_stop_model, fwd_node_model, fwd_edge_model, random_action_prob=0, apply_masks=True)
    bck_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], bck_stop_model, bck_node_model, bck_edge_model, random_action_prob=0, apply_masks=False)

    log_p_f = fwd_action_probs[list(range(len(fwd_action_probs))), actions]
    log_p_b = bck_action_probs[list(range(len(bck_action_probs))), torch.roll(actions, 1, 0)]  # prob of previous action
    log_p_b = torch.roll(log_p_b, -1, 0)  # we could save a roll here but it would be confusing

    final_graph_idxs = torch.cumsum(traj_lens, 0) - 1

    # don't count padding states (kind of wasteful to compute these)
    log_p_f[final_graph_idxs] = 0
    log_p_b[final_graph_idxs] = 0

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    tlm_traj_log_p_b = scatter(log_p_b.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

    with torch.no_grad():  # necessary?
        uniform_log_p_b = torch.tensor([-sum([math.log(get_num_previous_acts(s)) for s, _a in t[1:]]) for t in jagged_trajs], device=device)

    traj_log_p_b = (1-a) * tlm_traj_log_p_b + (a) * uniform_log_p_b

    back_loss = -traj_log_p_b.mean()  # this is kind of weird
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
        a=0.5,
        device="cuda"
    ):

    fwd_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], fwd_stop_model, fwd_node_model, fwd_edge_model, random_action_prob=0, apply_masks=True)
    bck_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], bck_stop_model, bck_node_model, bck_edge_model, random_action_prob=0, apply_masks=False)

    log_p_f = fwd_action_probs[list(range(len(fwd_action_probs))), actions]
    log_p_b = bck_action_probs[list(range(len(bck_action_probs))), torch.roll(actions, 1, 0)]  # prob of previous action
    log_p_b = torch.roll(log_p_b, -1, 0)  # we could save a roll here but it would be confusing

    final_graph_idxs = torch.cumsum(traj_lens, 0) - 1

    # don't count padding states (kind of wasteful to compute these)
    log_p_f[final_graph_idxs] = 0
    log_p_b[final_graph_idxs] = 0

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    tlm_traj_log_p_b = scatter(log_p_b.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

    back_loss = -tlm_traj_log_p_b.mean()
    tlm_traj_log_p_b = tlm_traj_log_p_b.detach()

    with torch.no_grad():  # necessary?
        uniform_log_p_b = torch.tensor([-sum([math.log(get_num_previous_acts(s)) for s, _a in t[1:]]) for t in jagged_trajs], device=device)

    traj_log_p_b = (1-a) * tlm_traj_log_p_b + (a) * uniform_log_p_b

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

    fwd_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], fwd_stop_model, fwd_node_model, fwd_edge_model, random_action_prob=0, apply_masks=True)
    bck_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], bck_stop_model, bck_node_model, bck_edge_model, random_action_prob=0, apply_masks=False)

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

@process_trajs
def get_tb_loss_frozen(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        fwd_stop_model, fwd_node_model, fwd_edge_model,
        bck_stop_model, bck_node_model, bck_edge_model,
        device="cuda"
    ):

    fwd_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], fwd_stop_model, fwd_node_model, fwd_edge_model, random_action_prob=0, apply_masks=True)
    bck_action_probs = get_action_probs(*raw_embeddings[1], *embedding_structure[1], bck_stop_model, bck_node_model, bck_edge_model, random_action_prob=0, apply_masks=False)

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

    traj_log_p_b = traj_log_p_b.detach()

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
                  "mean_num_nodes": mean_num_nodes.item(), "tb_loss": loss.detach()}

@process_trajs
def get_tb_loss_free(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        fwd_stop_model, fwd_node_model, fwd_edge_model,
        bck_stop_model, bck_node_model, bck_edge_model,
        device="cuda"
    ):

    fwd_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], fwd_stop_model, fwd_node_model, fwd_edge_model, random_action_prob=0, apply_masks=True)
    bck_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], bck_stop_model, bck_node_model, bck_edge_model, random_action_prob=0, apply_masks=False)

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
    loss = huber(traj_diffs).mean()

    connected_prop = mean_num_nodes = 0
    for traj in jagged_trajs:
        nodes, edges, _mask = traj[-2][0]
        num_nodes = torch.sum(torch.sum(nodes, dim=1) > 0, dim=0)
        num_edges = torch.sum(edges[:, :, 0], dim=(0, 1))
        connected_prop += (num_edges == num_nodes**2) / len(jagged_trajs)
        mean_num_nodes += num_nodes / len(jagged_trajs)

    return loss, {"log_z": log_z.item(), "mean_log_reward": torch.mean(log_rewards).item(), "connected_prop": connected_prop.item(),
                  "mean_num_nodes": mean_num_nodes.item(), "tb_loss": loss.detach()}

@process_trajs
def get_tb_loss_maxent(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        fwd_stop_model, fwd_node_model, fwd_edge_model,
        bck_stop_model, bck_node_model, bck_edge_model,
        n_model,
        device="cuda"
    ):

    fwd_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], fwd_stop_model, fwd_node_model, fwd_edge_model, random_action_prob=0, apply_masks=True)
    bck_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], bck_stop_model, bck_node_model, bck_edge_model, random_action_prob=0, apply_masks=False)

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

    traj_pred_l = n_model(raw_embeddings[0][2][embedding_structure[0][0]][final_graph_idxs - 1])
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
def _get_tb_loss_rand_const(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        stop_model, node_model, edge_model,
        mean=0.2, std=0.125, eps=0.01, precision=1_000, seed=1,
        device="cuda"
    ):

    action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], stop_model, node_model, edge_model, random_action_prob=0, apply_masks=True)

    log_p_f = action_probs[list(range(len(action_probs))), actions]
    unnorm_p_b = torch.tensor([get_state_hash((tuple(s), a, seed)) % precision for traj in jagged_trajs for (s, _a), (_s, a) in zip(traj, [traj[-1]] + traj[:-1])])
    log_p_b = torch.log(torch.clamp((unnorm_p_b / precision - 0.5) * (std * 2) + mean, min=eps, max=1))
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
                  "mean_num_nodes": mean_num_nodes.item(), "tb_loss": loss.detach()}

@process_trajs
def get_tb_loss_rand(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        stop_model, node_model, edge_model,
        std=0.125, eps=0.01,
        device="cuda"
    ):

    action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], stop_model, node_model, edge_model, random_action_prob=0, apply_masks=True)

    log_p_f = action_probs[list(range(len(action_probs))), actions]

    final_graph_idxs = torch.cumsum(traj_lens, 0) - 1

    # don't count padding states (kind of wasteful to compute these)
    log_p_f[final_graph_idxs] = 0

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = torch.tensor([-sum([math.log(get_num_previous_acts(s)) for s, _a in t[1:]]) for t in jagged_trajs], device=device)
    traj_log_p_b = torch.log(torch.clamp(torch.normal(mean=torch.exp(traj_log_p_b), std=std*(traj_lens-1), size=(action_probs.shape[0],)), min=eps, max=1))

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
                  "mean_num_nodes": mean_num_nodes.item(), "tb_loss": loss.detach()}

@process_trajs
def get_tb_loss_const(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        stop_model, node_model, edge_model,
        value=0.2,
        device="cuda"
    ):

    action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], stop_model, node_model, edge_model, random_action_prob=0, apply_masks=True)

    log_p_f = action_probs[list(range(len(action_probs))), actions]

    final_graph_idxs = torch.cumsum(traj_lens, 0) - 1

    # don't count padding states (kind of wasteful to compute these)
    log_p_f[final_graph_idxs] = 0

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = torch.full((traj_lens.shape[0],), math.log(value), device=device) * traj_lens

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
                  "mean_num_nodes": mean_num_nodes.item(), "tb_loss": loss.detach()}

@process_trajs
def get_tb_loss_aligned(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        stop_model, node_model, edge_model,
        correct_val=0.9, incorrect_val=0.02, reward_arg=0.8, reward_idx=1,
        device="cuda"
    ):  # aligns backward policy with handmade policy

    action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], stop_model, node_model, edge_model, random_action_prob=0, apply_masks=True)

    log_p_f = action_probs[list(range(len(action_probs))), actions]
    log_p_b = torch.tensor([
        get_aligned_action_log_prob(*s, a, reward_idx=reward_idx, reward_arg=reward_arg, correct_log_prob=math.log(correct_val), incorrect_log_prob=math.log(incorrect_val)) for traj in jagged_trajs for s, a in traj
    ])

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
                  "mean_num_nodes": mean_num_nodes.item(), "tb_loss": loss.detach()}

@process_trajs
def get_tb_loss_loss_aligned(
        jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
        actions, log_z, log_rewards,
        stop_model, node_model, edge_model,
        iters=5, std_mult=0.5, eps=0.01, device="cuda"
    ):

    action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], stop_model, node_model, edge_model, random_action_prob=0, apply_masks=True)

    log_p_f = action_probs[list(range(len(action_probs))), actions]
    log_p_f[torch.cumsum(traj_lens, 0) - 1] = 0  # don't count padding states (kind of wasteful to compute these)

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    uniform_traj_log_p_b = torch.tensor([-sum([math.log(get_num_previous_acts(s)) for s, _a in t[1:]]) for t in jagged_trajs], device=device)

    log_rewards = log_rewards.to(device)

    traj_log_p_b = torch.tensor(uniform_traj_log_p_b)

    for _ in range(iters):  # pray it converges

        inv_loss = 1 / huber((log_z + traj_log_p_f) - (log_rewards + traj_log_p_b))  # log_z gets broadcast into a vector here
        traj_log_p_b = (torch.log(torch.clamp(((inv_loss - inv_loss.mean()) / inv_loss.std() * std_mult + 1), min=eps, max=1)) + uniform_traj_log_p_b).detach()

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
                  "mean_num_nodes": mean_num_nodes.item(), "tb_loss": loss.detach()}



@torch.no_grad()
def get_action_log_probs_test_helper(nodes, edges, masks, loss_fn, base_model, fwd_models, bck_models, log_z_model, log_z, loss_arg_a, loss_arg_b, loss_arg_c, device="cuda"):
    # maybe worth implementing this for all losses and adding to configs dictionary

    raw_embeddings, embedding_structure = get_embeddings(base_model, nodes, edges, masks, device=device)
    log_z = log_z if log_z is not None else log_z_model(torch.tensor([[1.]], device=device))
    fwd_action_probs = get_action_probs(*raw_embeddings, *embedding_structure, *fwd_models, random_action_prob=0, apply_masks=True)

    if loss_fn == "tb-uniform-rand":
        unnorm_p_b = torch.tensor([[get_state_hash((s, a, loss_arg_c)) % 1_000 for a in range(s[0].shape[0]**2 + 2)] for s in zip(nodes, edges, masks, strict=True)])
        bck_action_probs = torch.log(torch.clamp((unnorm_p_b / 1_000 - 0.5) * (loss_arg_b * 2) + loss_arg_a, min=0.01, max=1))
    elif loss_fn == "tb-uniform-rand-var":
        log_p_b = torch.normal(mean=loss_arg_a, std=loss_arg_b, size=(nodes.shape[0],))
        bck_action_probs = torch.log(torch.clamp(log_p_b, min=0.01, max=1))
    elif loss_fn == "tb-uniform":
        bck_action_probs = torch.tensor([[-math.log(get_num_previous_acts(s))] * (s[0].shape[0]**2 + 2) for s in zip(nodes, edges, masks, strict=True)])
    elif loss_fn == "tb-tlm":
        bck_action_probs = get_action_probs(*raw_embeddings, *embedding_structure, *bck_models, random_action_prob=0, apply_masks=False)
    elif loss_fn == "tb-smoothed-tlm":
        tlm_log_p_b = get_action_probs(*raw_embeddings, *embedding_structure, *bck_models, random_action_prob=0, apply_masks=False)
        uniform_log_p_b = torch.tensor([[-math.log(get_num_previous_acts(s))] * (s[0].shape[0]**2 + 2) for s in zip(nodes, edges, masks, strict=True)])
        bck_action_probs = (1-loss_arg_a) * tlm_log_p_b + (loss_arg_a) * uniform_log_p_b
    else:
        bck_action_probs = None
        #raise ValueError(f"embedding for loss function {loss_fn} is not supported")
    
    return fwd_action_probs, bck_action_probs
