"""Equal client means (Eq. 6), CRA (Eqs. 11–12), plain baselines, and the Clustering baseline."""

import numpy as np
import torch

from clustering import complete_linkage_bisection, similarity_matrix
from defense import SMed, SCos, cosine, normalize_gradient


def cra_aggregate(cluster_updates, global_vector, server_eta=1.0, previous_update=None):
    gradients = np.asarray(cluster_updates, dtype=np.float64)
    # Reference: the previous global update; the global model in the first round.
    reference_vector = previous_update if previous_update is not None else global_vector
    reference_vector = reference_vector.detach().cpu().numpy()
    similarities = [cosine(gradient, reference_vector) for gradient in gradients]
    baseline = int(np.argmin(similarities))
    mu = np.asarray([1.0 - max(0.0, cosine(gradient, gradients[baseline]))
                     for gradient in gradients])
    if np.linalg.norm(gradients[baseline]) != 0:
        mu[baseline] = 0.0  # Its self-cosine is exactly one.
    fallback = bool(mu.sum() == 0)
    weights = np.full(len(mu), 1.0 / len(mu)) if fallback else mu / mu.sum()
    aggregate = torch.as_tensor(weights @ gradients, dtype=global_vector.dtype,
                                device=global_vector.device)
    return global_vector - server_eta * aggregate, {
        'mu_values': mu.tolist(), 'weights': weights.tolist(),
        'baseline_index': baseline, 'reference_similarities': similarities,
        'cra_fallback': fallback,
    }


def apfed_round(updates_by_client, clusters, g0_by_cluster, global_vector,
                args, pk=None, sk=None, previous_update=None):
    aggregates, contributing, detected, scores = [], [], [], {}
    for cluster_id, members in enumerate(clusters):
        leader = np.asarray(g0_by_cluster[cluster_id], dtype=np.float64)
        normalized = np.stack([normalize_gradient(updates_by_client[i], leader) for i in members])
        median = SMed(normalized, crypto=args.crypto, pk=pk, sk=sk, r=args.quant_bits)
        cluster_scores = [SCos(g, median, crypto=args.crypto, pk=pk, sk=sk, r=args.quant_bits)
                          for g in normalized]
        benign = []
        for index, client in enumerate(members):
            scores[client] = cluster_scores[index]
            if cluster_scores[index] >= args.cos_threshold:
                benign.append(index)
            else:
                detected.append(client)
        if benign:
            aggregates.append(normalized[benign].mean(axis=0))
            contributing.append(cluster_id)
    details = {'detected_malicious_ids': sorted(detected), 'cosines': scores,
               'contributing_cluster_ids': contributing}
    if not aggregates:
        details.update(weights=[], mu_values=[], baseline_index=None,
                       reference_similarities=[], cra_fallback=False, skipped_round=True)
        return global_vector.clone(), details
    new_vector, cra = cra_aggregate(aggregates, global_vector, args.server_eta, previous_update)
    details.update(cra, skipped_round=False)
    details['baseline_cluster_id'] = (None if cra['baseline_index'] is None
                                      else contributing[cra['baseline_index']])
    return new_vector, details


def baseline_round(updates, global_vector, defense, ratio=0.0, server_eta=1.0):
    values = np.asarray(updates, dtype=np.float64)
    if defense == 'fedavg':
        aggregate = values.mean(axis=0)
    elif defense == 'median':
        aggregate = np.median(values, axis=0)
    elif defense == 'trimmed':
        trim = int(ratio * len(values))
        if 2 * trim >= len(values):
            raise ValueError('trimmed mean requires 2 * floor(ratio * num_users) < num_users')
        aggregate = np.sort(values, axis=0)[trim:len(values) - trim].mean(axis=0)
    else:
        raise ValueError('unknown baseline: ' + defense)
    return global_vector - server_eta * torch.as_tensor(
        aggregate, dtype=global_vector.dtype, device=global_vector.device)


def clustering_round(updates, global_vector, server_eta=1.0):
    """Clustering baseline: one cosine bisection of the uploads, mean of the larger cluster.

    The cut is the complete-linkage split of Sattler et al.'s reference code. Equal
    halves have no majority, so no client is rejected. Returns the rejected indices too.
    """
    values = np.asarray(updates, dtype=np.float64)
    kept, rejected = list(range(len(values))), []
    if len(values) > 1:
        left, right, _ = complete_linkage_bisection(similarity_matrix(values))
        if len(left) != len(right):
            kept, rejected = (left, right) if len(left) > len(right) else (right, left)
    aggregate = values[kept].mean(axis=0)
    return global_vector - server_eta * torch.as_tensor(
        aggregate, dtype=global_vector.dtype, device=global_vector.device), rejected
