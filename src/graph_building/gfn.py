import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from util import get_num_previous_acts, adjust_action_idxs, huber


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

def trajs_to_tensors(trajs):

    # pad inputs to the same length (this is quite memory intensive)
    nodes = nn.utils.rnn.pad_sequence([n for (n, _e, _m), _a in trajs], batch_first=True)
    edges = torch.stack([F.pad(e, (0, 0, 0, nodes.shape[1] - e.shape[1], 0, nodes.shape[1] - e.shape[0]), "constant", 0) for (_n, e, _m), _a in trajs])
    masks = torch.stack([F.pad(m, (0, nodes.shape[1] - m.shape[0]), "constant", 0) for (_n, _e, m), _a in trajs])

    pre_padding_lens = [n.shape[0] for (n, _e, _m), _a in trajs]
    post_padding_len = nodes.shape[1]
    actions = adjust_action_idxs(torch.tensor([a for _s, a in trajs]), pre_padding_lens, post_padding_len)

    return nodes, edges, masks, actions

def process_trajs(get_loss_fn):  # this wrapper is so unnecessary but is annoying to get rid of

    def get_loss_fn_from_trajs(jagged_trajs, log_rewards, base_models, log_z_model, *model_heads, constant_log_z=None, device="cuda", **kwargs):

        for m in (*base_models, *model_heads):
            m.train()

        # flatten inputs from jagged list
        traj_lens = torch.tensor([len(t) for t in jagged_trajs], device=device)
        trajs = [action for traj in jagged_trajs for action in traj]

        nodes, edges, masks, actions = trajs_to_tensors(trajs)

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

def get_get_tb_loss_backward(get_bck_probs, get_bck_loss):

    @process_trajs
    def get_tb_loss_backward(
            jagged_trajs, traj_lens, raw_embeddings, embedding_structure,
            actions, log_z, log_rewards,
            fwd_stop_model, fwd_node_model, fwd_edge_model,
            *bck_models, device="cuda", **kwargs
        ):

        trajs = [action for traj in jagged_trajs for action in traj]

        fwd_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], fwd_stop_model, fwd_node_model, fwd_edge_model, random_action_prob=0, apply_masks=True)
        log_p_f = fwd_action_probs[list(range(len(fwd_action_probs))), actions]

        log_p_b, info, bck_metrics = get_bck_probs(trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models, **kwargs)  # prob of previous action
        log_p_b = torch.roll(log_p_b, -1, 0)

        final_graph_idxs = torch.cumsum(traj_lens, 0) - 1

        # don't count padding states (kind of wasteful to compute these)
        log_p_f[final_graph_idxs] = 0
        log_p_b[final_graph_idxs] = 0

        batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
        traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
        traj_log_p_b = scatter(log_p_b.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

        log_rewards = log_rewards.to(device)
        bck_loss, replacement_backward = get_bck_loss(log_z, traj_log_p_f, log_rewards, traj_log_p_b, info, **kwargs)
        traj_log_p_b = traj_log_p_b.detach()

        if replacement_backward is not None:  # bit hacky
            traj_log_p_b = replacement_backward

        traj_diffs = (log_z + traj_log_p_f) - (log_rewards + traj_log_p_b)  # log_z gets broadcast into a vector here

        tb_loss = huber(traj_diffs)

        loss = tb_loss.mean() + bck_loss.mean()

        return loss, bck_metrics | {"log_z": log_z.item(), "loss": loss.detach(), "tb_loss": tb_loss.detach(), "bck_loss": bck_loss.detach()}

    return get_tb_loss_backward

@torch.no_grad()
def get_metrics(nodes, edges, masks, actions, traj_lens, log_rewards,
                base_models, fwd_models, bck_models, constant_log_z, log_z_model,
                get_bck_probs, get_bck_loss, device="cuda", **kwargs):

    log_z = torch.tensor(constant_log_z) if constant_log_z is not None else log_z_model(torch.tensor([[1.]], device=device))

    embeddings = []
    for base_model in base_models:
        embeddings.append(get_embeddings(base_model, nodes, edges, masks, device=device))
    raw_embeddings, embedding_structure = list(zip(*embeddings))

    fwd_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], *fwd_models, random_action_prob=0, apply_masks=True)
    log_p_f = fwd_action_probs[list(range(len(fwd_action_probs))), actions]

    log_p_b, info, bck_metrics = get_bck_probs(list(zip(nodes, edges, masks)), traj_lens, actions, raw_embeddings, embedding_structure, bck_models, **kwargs)  # prob of previous action
    log_p_b = torch.roll(log_p_b, -1, 0)

    final_graph_idxs = torch.cumsum(traj_lens, 0) - 1

    # don't count padding states (kind of wasteful to compute these)
    log_p_f[final_graph_idxs] = 0
    log_p_b[final_graph_idxs] = 0

    batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)
    traj_log_p_f = scatter(log_p_f.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")
    traj_log_p_b = scatter(log_p_b.to(device), batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

    log_rewards = log_rewards.to(device)
    bck_loss, replacement_backward = get_bck_loss(log_z, traj_log_p_f, log_rewards, traj_log_p_b, info, **kwargs)
    traj_log_p_b = traj_log_p_b.detach()

    if replacement_backward is not None:  # bit hacky
        traj_log_p_b = replacement_backward

    bck_flow = log_rewards + traj_log_p_b
    traj_diffs = (log_z + traj_log_p_f) - bck_flow  # log_z gets broadcast into a vector here

    tb_loss = huber(traj_diffs)

    loss = tb_loss + bck_loss

    return (bck_metrics | {
        "fwd_prob_mean": log_p_f.mean().item(),
        "fwd_prob_std": log_p_f.std().item(),
        "bck_prob_mean": log_p_b.mean().item(),
        "bck_prob_std": log_p_b.std().item(),
        "traj_fwd_prob_mean": traj_log_p_f.mean().item(),
        "traj_fwd_prob_std": traj_log_p_f.std().item(),
        "traj_bck_prob_mean": traj_log_p_b.mean().item(),
        "traj_bck_prob_std": traj_log_p_b.std().item(),
        "log_z": log_z.item(),
        "bck_flow_mean": bck_flow.mean().item(),
        "bck_flow_std": bck_flow.std().item(),
        "fwd_loss_mean": tb_loss.mean().item(),
        "fwd_loss_std": tb_loss.std().item(),
        "combined_loss_mean": loss.mean().item(),
        "combined_loss_std": loss.std().item(),
        "bck_loss_mean": bck_loss.mean().item(),
        "bck_loss_std": bck_loss.std().item()
    }, fwd_action_probs, log_p_b)  # not worth it to predict probabilities of backward actions as it is expensive and we often don't have to
