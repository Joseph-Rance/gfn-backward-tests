import math
import torch
from torch_scatter import scatter

from gfn import get_action_probs
from util import get_num_previous_acts, is_n_connected, get_aligned_action_log_prob, get_state_hash, huber


def get_bck_probs_const(trajs, _traj_lens, _actions, _raw_embeddings, _embedding_structure, _bck_models, value=0.2, **_kwargs):
    return torch.full((len(trajs),), math.log(value)), None, {"mean_single_state_p_b_std": 0, "std_single_state_p_b_std": 0}

def get_bck_probs_uniform(trajs, _traj_lens, _actions, _raw_embeddings, _embedding_structure, _bck_models, **_kwargs):
    return torch.tensor([-math.log(get_num_previous_acts(s)) for s, _a in trajs]), None, {"mean_single_state_p_b_std": 0, "std_single_state_p_b_std": 0}

def get_bck_probs_action_mult(trajs, _traj_lens, _actions, _raw_embeddings, _embedding_structure, _bck_models, action_type=0, n=1, **_kwargs):

    single_state_p_b_std = []

    log_p_b = torch.zeros((len(trajs),))
    for i, ((__, a), (s, __)) in enumerate(zip(trajs[:-1], trajs[1:])):
        num = get_num_previous_acts(s)

        if action_type == 0:  # edge
            num_edges = torch.sum(s[1][:, :, 0], dim=(0, 1))
            if a < len(s[2])**2:  # if action is add edge, assign more weight
                log_p_b[i+1] = math.log(n/(num+num_edges*(n-1)))
            else:  # otherwise, assign normal weight
                log_p_b[i+1] = math.log(1/(num+num_edges*(n-1)))  # we still need to adjust for the added weight of a potential add edge

            single_state_p_b_std.append(torch.tensor([n]*num_edges + [1]*(num-num_edges)).std())  # slow but oh well

        elif action_type == 1:  # node
            if a == len(s[2])**2:  # if action is add node, assign more weight
                log_p_b[i+1] = math.log(n/(num+n-1))
            else:  # otherwise, assign normal weight
                num_nodes = torch.sum(torch.sum(s[0], dim=1) > 0, dim=1)
                has_disconnected = int((torch.sum(s[1][:, num_nodes-1]) == 0 and torch.sum(s[1][num_nodes-1, :]) == 0).item())
                log_p_b[i+1] = math.log(1/(num+has_disconnected*(n-1)))  # we need to adjust for the added weight of a potential add node

            single_state_p_b_std.append(torch.tensor([n] + [1]*(num-1)).std())

        else:  # stop
            assert action_type == 2
            if a == len(s[2])**2 + 1:  # if action is stop, assign more weight
                log_p_b[i+1] = math.log(n/(num+n-1))
            else:  # otherwise, assign normal weight (no need to adjust for a potential use of stop before a non-terminal state)
                log_p_b[i+1] = math.log(1/num)

            single_state_p_b_std.append(torch.tensor([n] + [1]*(num-1)).std())

        single_state_p_b_std = torch.tensor(single_state_p_b_std)

        return log_p_b, None, {"mean_single_state_p_b_std": single_state_p_b_std.mean().item(), "std_single_state_p_b_std": single_state_p_b_std.std().item()}

def get_bck_probs_rand(trajs, traj_lens, _actions, _raw_embeddings, _embedding_structure, _bck_models, std=0.125, eps=0.01, **_kwargs):
    uniform_p_b = torch.tensor([1/get_num_previous_acts(s) for s, _a in trajs])
    return (
        torch.log(torch.clamp(torch.normal(mean=uniform_p_b, std=std), min=eps, max=1)),
        None,
        {"mean_single_state_p_b_std": std, "std_single_state_p_b_std": 0}
    )

def get_bck_probs_aligned(trajs, _traj_lens, _actions, _raw_embeddings, _embedding_structure, _bck_models,
                          cor_val=0.9, inc_val=0.02, reward_arg=0.8, reward_idx=1, **_kwargs):
    # aligns backward policy with handmade policy
    log_p_b = torch.tensor([
        get_aligned_action_log_prob(*s, a, reward_idx=reward_idx, reward_arg=reward_arg,
                                    correct_log_prob=math.log(cor_val), incorrect_log_prob=math.log(inc_val)) for s, a in trajs
    ])
    return torch.roll(log_p_b, 1, 0), None, {"mean_single_state_p_b_std": -1, "std_single_state_p_b_std": -1}  # expensive to compute

def get_bck_probs_frozen(_trajs, _traj_lens, actions, raw_embeddings, embedding_structure, bck_models, **_kwargs):
    bck_action_probs = get_action_probs(*raw_embeddings[1], *embedding_structure[1], *bck_models, random_action_prob=0, apply_masks=False)
    single_state_p_b_std = torch.std(bck_action_probs, dim=1)
    return (
        bck_action_probs[list(range(len(bck_action_probs))), torch.roll(actions, 1, 0)],
        None,
        {"mean_single_state_p_b_std": single_state_p_b_std.mean().item(), "std_single_state_p_b_std": single_state_p_b_std.std().item()}
    )

def get_bck_probs_tlm(_trajs, _traj_lens, actions, raw_embeddings, embedding_structure, bck_models, **kwargs):
    bck_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], *bck_models, random_action_prob=0, apply_masks=False)
    single_state_p_b_std = torch.std(bck_action_probs, dim=1)
    return (
        bck_action_probs[list(range(len(bck_action_probs))), torch.roll(actions, 1, 0)],
        None,
        {"mean_single_state_p_b_std": single_state_p_b_std.mean().item(), "std_single_state_p_b_std": single_state_p_b_std.std().item()}
    )

def get_bck_loss_free(_log_z, traj_log_p_f, _log_rewards, traj_log_p_b, _info, **_kwargs):
    return torch.tensor([0]*len(traj_log_p_f)), traj_log_p_b

def get_bck_loss_tlm(_log_z, _traj_log_p_f, _log_rewards, traj_log_p_b, _info, **_kwargs):
    return -traj_log_p_b, None

def get_bck_probs_soft_tlm(trajs, _traj_lens, actions, raw_embeddings, embedding_structure, bck_models, a=0.5, **_kwargs):
    bck_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], *bck_models, random_action_prob=0, apply_masks=False)
    single_state_p_b_std = torch.std(bck_action_probs, dim=1)
    tlm_log_p_b = bck_action_probs[list(range(len(bck_action_probs))), torch.roll(actions, 1, 0)]
    with torch.no_grad():  # necessary?
        uniform_log_p_b = torch.tensor([-math.log(get_num_previous_acts(s)) for s, _a in trajs])
    return (
        (1-a) * tlm_log_p_b + (a) * uniform_log_p_b, tlm_log_p_b,
        None,
        {"mean_single_state_p_b_std": single_state_p_b_std.mean().item() * (1-a), "std_single_state_p_b_std": single_state_p_b_std.std().item() * (1-a)}
    )

def get_bck_probs_smooth_tlm(trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models, a=0.5, **_kwargs):
    bck_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], *bck_models, random_action_prob=0, apply_masks=False)
    single_state_p_b_std = torch.std(bck_action_probs, dim=1)
    tlm_log_p_b = bck_action_probs[list(range(len(bck_action_probs))), torch.roll(actions, 1, 0)]
    with torch.no_grad():  # necessary?
        uniform_log_p_b = torch.tensor([-math.log(get_num_previous_acts(s)) for s, _a in trajs])

    log_p_b = torch.roll(tlm_log_p_b, -1, 0)
    log_p_b[torch.cumsum(traj_lens, 0) - 1] = 0
    batch_idx = torch.arange(len(traj_lens)).repeat_interleave(traj_lens)
    traj_log_p_b = scatter(log_p_b, batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

    return (
        (1-a) * tlm_log_p_b + (a) * uniform_log_p_b, tlm_log_p_b,
        traj_log_p_b,
        {"mean_single_state_p_b_std": single_state_p_b_std.mean().item() * (1-a), "std_single_state_p_b_std": single_state_p_b_std.std().item() * (1-a)}
    )

def get_bck_loss_smooth_biased_tlm(_log_z, _traj_log_p_f, _log_rewards, _traj_log_p_b, info, **_kwargs):
    return -info, None

def get_bck_probs_biased_tlm(trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models, multiplier=5, ns=[3, 4, 5], **_kwargs):
    # bias tlm backward policy towards fully connected graph with nodes counts in ns
    bck_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], *bck_models, random_action_prob=0, apply_masks=False)
    tlm_log_p_b = bck_action_probs[list(range(len(bck_action_probs))), torch.roll(actions, 1, 0)]
    final_graphs = [trajs[idx][0] for idx in torch.cumsum(traj_lens, 0) - 2]
    biased_tlm_log_p_b = tlm_log_p_b * (1 + (multiplier - 1) * torch.tensor([int(is_n_connected(*g, ns=ns)) for g in final_graphs])).repeat_interleave(traj_lens)

    log_p_b = torch.roll(tlm_log_p_b, -1, 0)
    log_p_b[torch.cumsum(traj_lens, 0) - 1] = 0
    batch_idx = torch.arange(len(traj_lens)).repeat_interleave(traj_lens)
    traj_log_p_b = scatter(log_p_b, batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

    return biased_tlm_log_p_b, traj_log_p_b, {"mean_single_state_p_b_std": -1, "std_single_state_p_b_std": -1}  # expensive to compute

def get_bck_probs_max_ent(_trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models, **_kwargs):

    final_graph_idxs = torch.cumsum(traj_lens, 0) - 1
    first_graph_idxs = torch.roll(final_graph_idxs, 1, dims=0) + 1
    first_graph_idxs[0] = 0

    bck_action_probs = get_action_probs(*raw_embeddings[0], *embedding_structure[0], *bck_models, random_action_prob=0, apply_masks=False)
    single_state_p_b_std = torch.std(bck_action_probs, dim=1)
    log_p_b = bck_action_probs[list(range(len(bck_action_probs))), torch.roll(actions, 1, 0)]

    traj_pred_l = bck_models[-1](raw_embeddings[0][2][embedding_structure[0][0]][final_graph_idxs - 1])
    traj_pred_l[first_graph_idxs >= final_graph_idxs - 1] = 0

    return log_p_b, traj_pred_l, {"mean_single_state_p_b_std": single_state_p_b_std.mean().item(), "std_single_state_p_b_std": single_state_p_b_std.std().item()}

def get_bck_loss_max_ent(_log_z, _traj_log_p_f, _log_rewards, traj_log_p_b, traj_pred_l, **_kwargs):
    return huber(traj_log_p_b + traj_pred_l), None

def get_bck_loss_loss_aligned(log_z, traj_log_p_f, log_rewards, uniform_traj_log_p_b, _info, iters=5, std_mult=0.5, eps=0.01, **_kwargs):
    traj_log_p_b = torch.tensor(uniform_traj_log_p_b)
    for _ in range(iters):  # pray it converges
        print("does it converge?")
        inv_loss = 1 / huber((log_z + traj_log_p_f) - (log_rewards + traj_log_p_b))  # log_z gets broadcast into a vector here
        traj_log_p_b = (torch.log(torch.clamp(((inv_loss - inv_loss.mean()) / inv_loss.std() * std_mult + 1), min=eps, max=1)) + uniform_traj_log_p_b).detach()
    return torch.tensor([0]*len(traj_log_p_f)), traj_log_p_b.detach()

def get_bck_probs_meta(trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models, weights=None, reward_arg=0.8, **_kwargs):
    assert weights is not None

    c_0_v, __, c_0_m = get_bck_probs_const(trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models, value=0.15)
    u_0_v, __, u_0_m = get_bck_probs_uniform(trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models)
    a_0_v, __, a_0_m = get_bck_probs_action_mult(trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models, action_type=0, n=2)
    a_1_v, __, a_1_m = get_bck_probs_action_mult(trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models, action_type=0, n=0.5)
    a_2_v, __, a_2_m = get_bck_probs_action_mult(trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models, action_type=1, n=2)
    a_3_v, __, a_3_m = get_bck_probs_action_mult(trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models, action_type=1, n=0.5)
    a_4_v, __, a_4_m = get_bck_probs_action_mult(trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models, action_type=2, n=2)
    a_5_v, __, a_5_m = get_bck_probs_action_mult(trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models, action_type=2, n=0.5)
    r_0_v, __, r_0_m = get_bck_probs_rand(trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models, std=0.05)
    l_0_v, __, l_0_m = get_bck_probs_aligned(trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models, reward_arg=reward_arg)
    t_0_v, __, t_0_m = get_bck_probs_tlm(trajs, traj_lens, actions, raw_embeddings, embedding_structure, bck_models)

    log_p_b = torch.roll(t_0_v, -1, 0)
    log_p_b[torch.cumsum(traj_lens, 0) - 1] = 0
    batch_idx = torch.arange(len(traj_lens)).repeat_interleave(traj_lens)
    traj_log_p_b = scatter(log_p_b, batch_idx, dim=0, dim_size=traj_lens.shape[0], reduce="sum")

    joint_metrics = {"mean_single_state_p_b_std": 0, "std_single_state_p_b_std": 0}
    for w, m in zip(weights, [c_0_m, u_0_m, a_0_m, a_1_m, a_2_m, a_3_m, a_4_m, a_5_m, r_0_m, l_0_m, t_0_m]):
        joint_metrics["mean_single_state_p_b_std"] += c_0_m["mean_single_state_p_b_std"] * w
        joint_metrics["std_single_state_p_b_std"] += c_0_m["std_single_state_p_b_std"] * w

    back_probs = torch.transpose(torch.stack([c_0_v, u_0_v, a_0_v, a_1_v, a_2_v, a_3_v, a_4_v, a_5_v, r_0_v, l_0_v, t_0_v]))
    return torch.matmul(back_probs, weights), traj_log_p_b, joint_metrics

def _get_bck_probs_rand_const(jagged_trajs, _traj_lens, _actions, _raw_embeddings, _embedding_structure, _bck_models,
                              mean=0.2, std=0.125, eps=0.01, precision=1_000, seed=1, **_kwargs):
    unnorm_p_b = torch.tensor([get_state_hash((tuple(s), a, seed)) % precision for traj in jagged_trajs for (s, _a), (_s, a) in zip(traj, [traj[-1]] + traj[:-1])])
    log_p_b = torch.log(torch.clamp((unnorm_p_b / precision - 0.5) * (std * 2) + mean, min=eps, max=1))
    log_p_b = torch.roll(log_p_b, -1, 0)


const = (get_bck_probs_const, lambda *pargs, **_kwargs: (torch.tensor([0.]*pargs[1].shape[0], device=pargs[0].device), None))
uniform = (get_bck_probs_uniform, lambda *pargs, **_kwargs: (torch.tensor([0.]*pargs[1].shape[0], device=pargs[0].device), None))
action_mult = (get_bck_probs_action_mult, lambda *pargs, **_kwargs: (torch.tensor([0.]*pargs[1].shape[0], device=pargs[0].device), None))
rand = (get_bck_probs_rand, lambda *pargs, **_kwargs: (torch.tensor([0.]*pargs[1].shape[0], device=pargs[0].device), None))
aligned = (get_bck_probs_const, lambda *pargs, **_kwargs: (torch.tensor([0.]*pargs[1].shape[0], device=pargs[0].device), None))
frozen = (get_bck_probs_frozen, lambda *pargs, **_kwargs: (torch.tensor([0.]*pargs[1].shape[0], device=pargs[0].device), None))
free = (get_bck_probs_tlm, get_bck_loss_free)
tlm = (get_bck_probs_tlm, get_bck_loss_tlm)
soft_tlm = (get_bck_probs_soft_tlm, get_bck_loss_tlm)
smooth_tlm = (get_bck_probs_smooth_tlm, get_bck_loss_smooth_biased_tlm)
biased_tlm = (get_bck_probs_biased_tlm, get_bck_loss_smooth_biased_tlm)
max_ent = (get_bck_probs_max_ent, get_bck_loss_max_ent)
loss_aligned = (get_bck_probs_uniform, get_bck_loss_loss_aligned)  # single_state_p_b_std fails here
meta = (get_bck_probs_meta, get_bck_loss_smooth_biased_tlm)
