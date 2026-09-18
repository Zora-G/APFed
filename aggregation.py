import copy

import numpy as np
import torch

from defense import normalize_gradient


def _is_state_dict(values):
    return len(values) > 0 and isinstance(values[0], dict)

def _tensor(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    return torch.as_tensor(value, dtype=torch.float32)

def FedAvg(weights):
    """Equal-weight averaging of client updates."""
    if len(weights) == 0:
        raise ValueError('FedAvg needs at least one update')
    if _is_state_dict(weights):
        result = copy.deepcopy(weights[0])
        for key in result.keys():
            result[key] = sum(w[key] for w in weights) / len(weights)
        return result
    return sum((_tensor(w) for w in weights)) / float(len(weights))

def cluster_aggregate(updates, benign_indices=None, weights=None):
    """LN aggregation of retained gradients g_i=theta_global-theta_local."""
    if benign_indices is None:
        benign_indices = list(range(len(updates)))
    kept = [updates[i] for i in benign_indices]
    if len(kept) == 0:
        return None
    if weights is None:
        return FedAvg(kept)
    total = float(sum(weights[i] for i in benign_indices))
    if total == 0:
        return FedAvg(kept)
    if _is_state_dict(kept):
        result = copy.deepcopy(kept[0])
        for key in result.keys():
            result[key] = sum(updates[i][key] * (weights[i] / total)
                              for i in benign_indices)
        return result
    return sum((_tensor(updates[i]) * (weights[i] / total)
                for i in benign_indices))


def _as_numpy(updates):
    return np.stack([x.detach().cpu().numpy().astype(float) for x in updates])

def _assert_finite_update(value, *, location, round_index=None, client_id=None,
                          cluster_index=None):
    """Check updates before quantization."""
    if hasattr(value, "detach"):
        checked = value.detach().cpu().float().reshape(-1)
        finite = torch.isfinite(checked)
        count = int(checked.numel())
        if bool(finite.all()):
            return
        bad = int((~finite).nonzero(as_tuple=False)[0].item())
        bad_value = float(checked[bad].item())
    else:
        checked = np.asarray(value, dtype=float).reshape(-1)
        finite = np.isfinite(checked)
        count = int(checked.size)
        if bool(np.all(finite)):
            return
        bad = int(np.flatnonzero(~finite)[0])
        bad_value = float(checked[bad])
    context = ["non-finite value", "location={}".format(location),
               "coordinate={}/{}".format(bad, count),
               "value={!r}".format(bad_value)]
    if round_index is not None:
        context.append("round={}".format(int(round_index)))
    if client_id is not None:
        context.append("client_id={}".format(int(client_id)))
    if cluster_index is not None:
        context.append("cluster_index={}".format(int(cluster_index)))
    raise FloatingPointError("; ".join(context))

def _normalize_cluster(grads, g_0, root_norm=0.0):
    grads = np.asarray(grads, dtype=float)
    g_0 = np.asarray(g_0, dtype=float).reshape(-1).copy()
    norm_0 = np.linalg.norm(g_0)
    if root_norm > 0 and norm_0 > 0:
        g_0 = g_0 * (float(root_norm) / norm_0)
    if np.linalg.norm(g_0) == 0:
        return grads.copy()
    return np.stack([normalize_gradient(g_i, g_0) for g_i in grads])

def apfed_round(updates_by_client, fixed_clusters, g0_by_cluster,
                  global_vector, args, guard, pk=None, sk=None, round_index=None):
    cluster_updates = []
    valid_cluster_ids = []
    active_clusters = []
    invalid_clusters = []
    detected_clients = []
    scores_by_client = {}

    for cluster_index, cluster in enumerate(fixed_clusters):
        participant_ids = [client_id for client_id in cluster
                           if client_id in updates_by_client]
        if len(participant_ids) == 0:
            continue
        active_clusters.append(tuple(participant_ids))
        raw = _as_numpy([updates_by_client[client_id]
                         for client_id in participant_ids])
        _assert_finite_update(raw, location="apfed_cluster_raw",
                              round_index=round_index,
                              cluster_index=cluster_index)
        _assert_finite_update(g0_by_cluster[cluster_index],
                              location="apfed_leader_gradient",
                              round_index=round_index,
                              cluster_index=cluster_index)
        grads = _normalize_cluster(
            raw, g0_by_cluster[cluster_index], args.root_norm)
        _assert_finite_update(grads, location="apfed_cluster_normalized",
                              round_index=round_index,
                              cluster_index=cluster_index)
        benign, malicious, scores = guard.filter(
            grads, args=args, crypto=args.crypto, pk=pk, sk=sk,
            cos_threshold=args.cos_threshold, r=args.quant_bits)
        for local_index, score in enumerate(scores):
            scores_by_client[participant_ids[local_index]] = float(score)
        detected_clients.extend(participant_ids[i] for i in malicious)
        aggregate = cluster_aggregate(grads, benign)
        if aggregate is None:
            invalid_clusters.append(cluster_index)
            continue
        cluster_updates.append(aggregate)
        valid_cluster_ids.append(cluster_index)

    if len(cluster_updates) == 0:
        cra_details = {
            'mu_values': [],
            'normalized_weights': [],
            'baseline_index': None,
            'baseline_cluster_id': None,
            'baseline_similarity_to_previous_global': None,
            'cra_zero_weight_fallback': False,
            'input_type': 'gradient',
            'mode': args.cra_mode,
        }
        return (global_vector.detach().clone(), active_clusters,
                detected_clients, scores_by_client, cra_details,
                invalid_clusters, True, 0)

    new_vector, cra_details = guard.cra(
        cluster_updates, global_vector,
        eta=getattr(args, 'cra_eta', args.cra_lr),
        mode=args.cra_mode, input_type='gradient',
        cluster_ids=valid_cluster_ids)
    return (new_vector, active_clusters, detected_clients, scores_by_client,
            cra_details, invalid_clusters, False,
            len(cluster_updates))
