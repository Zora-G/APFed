"""One-shot CFL: cosine affinities and exact minimax bisection (Eqs. 4–5)."""

import numpy as np


def similarity_matrix(updates):
    vectors = np.asarray(updates, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
    return np.clip(unit @ unit.T, -1.0, 1.0)


def mst_bisection(affinity):
    """Return local indices of the optimal cut and its maximum cross affinity.

    Dense maximum-Prim takes O(n²). Every cut crosses a tree edge, so its
    bottleneck is at least the tree's minimum edge. Removing that edge attains
    the bound by the maximum-spanning-tree exchange property.
    """
    affinity = np.asarray(affinity)
    n = len(affinity)
    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    best = affinity[0].copy()
    parent = np.zeros(n, dtype=int)
    edges = []
    for _ in range(n - 1):
        node = int(np.argmax(np.where(selected, -np.inf, best)))
        edges.append((int(parent[node]), node, float(best[node])))
        selected[node] = True
        improved = (~selected) & (affinity[node] > best)
        best[improved] = affinity[node, improved]
        parent[improved] = node
    cut = min(range(len(edges)), key=lambda i: edges[i][2])
    adjacency = [[] for _ in range(n)]
    for index, (a, b, _) in enumerate(edges):
        if index != cut:
            adjacency[a].append(b)
            adjacency[b].append(a)
    reached = {0}
    pending = [0]
    while pending:
        node = pending.pop()
        for other in adjacency[node]:
            if other not in reached:
                reached.add(other)
                pending.append(other)
    return sorted(reached), sorted(set(range(n)) - reached), edges[cut][2]


def cfl_clusters(updates, threshold=0.5, min_size=1, max_clusters=0):
    """Recursively split in depth-first order, keeping the W0 partition fixed."""
    affinity = similarity_matrix(updates)
    clusters = [list(range(len(updates)))]
    index = 0
    while index < len(clusters):
        cluster = clusters[index]
        if max_clusters and len(clusters) >= max_clusters:
            break
        if len(cluster) >= 2 * min_size:
            left, right, cross = mst_bisection(affinity[np.ix_(cluster, cluster)])
            if cross < threshold and min(len(left), len(right)) >= min_size:
                clusters[index:index + 1] = [
                    [cluster[i] for i in left], [cluster[i] for i in right]]
                continue
        index += 1
    return clusters


def complete_linkage_bisection(affinity):
    """Two clusters by complete-linkage agglomeration, and the cut's maximum cross affinity.

    Clusters merge in order of their largest minimum pairwise affinity until two
    remain, as in the reference implementation of Sattler et al. (complete linkage
    on the negated cosine matrix). The first maximum breaks ties.
    """
    affinity = np.asarray(affinity, dtype=np.float64)
    members = [[i] for i in range(len(affinity))]
    lowest = affinity.copy()
    np.fill_diagonal(lowest, -np.inf)
    while len(members) > 2:
        a, b = divmod(int(np.argmax(lowest)), len(lowest))
        a, b = min(a, b), max(a, b)
        merged = np.minimum(lowest[a], lowest[b])
        lowest[a], lowest[:, a] = merged, merged
        lowest[a, a] = -np.inf
        lowest = np.delete(np.delete(lowest, b, axis=0), b, axis=1)
        members[a] = members[a] + members.pop(b)
    left, right = sorted(members[0]), sorted(members[1])
    return left, right, float(affinity[np.ix_(left, right)].max())
