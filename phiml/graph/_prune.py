from typing import Callable

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, csc_matrix

from ..math import math, Tensor, wrap, channel, scatter, primal, dual, concat, stored_indices, instance, stored_values, to_format, dsum, ones, tensor_like, nonzero


def eliminate_skips(sparse_distances: Tensor, threshold=1-1e-3, threshold_sum=1.5, eliminate_zeros=True):
    """
    Eliminate skip connections in the graph, i.e. edges between nodes where there is a closer node connected to both.

    Edges AB are skip connections if there exists a node C connected to both A and B with `AC and BC < threshold * AB` and `sum(AC,BC) < threshold_sum * AB`.

    Args:
        sparse_distances: Dense or sparse symmetric matrix. Graph with edges storing the distances between nodes. Distances of 0 are taken to be not connected.
        threshold: Distance factor to decide which edges to drop based on individual distances.
        threshold_sum: Distance factor to decide which edges to drop based on the sum of distances.
        eliminate_zeros: Whether to also drop all zero-distance edges.
    """
    np_dist = sparse_distances.numpy([primal, dual])
    if isinstance(np_dist, (coo_matrix, csr_matrix, csc_matrix)):
        np_dist = np_dist.copy()
        np_dist.eliminate_zeros()
    lil = lil_matrix(np_dist)
    to_break = []
    for a, (neighbors, dists) in enumerate(zip(lil.rows, lil.data)):
        for b, ab in zip(neighbors, dists):
            for c, ac in zip(neighbors, dists):
                if b != c:
                    if b in lil.rows[c]:  # C connects to A and B
                        bc = lil.data[c][lil.rows[c].index(b)]
                        if ac + bc < threshold_sum * ab and ac < threshold * ab and bc < threshold * ab:
                            to_break.append((a, b))
    to_break = wrap(to_break, 'pairs:i', channel(index=sparse_distances.shape.names))
    if eliminate_zeros:
        indices = stored_indices(sparse_distances, instance('pairs'))[channel(to_break).labels[0]]
        values = stored_values(sparse_distances, instance('pairs'))
        zero_positions = indices[values == 0]
        to_break = concat([to_break, zero_positions], 'pairs')
    return scatter(sparse_distances, to_break, 0, outside_handling='undefined')


def min_valid_edge_count_mask(graph: Tensor, directed=False, min_edges=2, mask: Tensor = None):
    assert not directed, f"directed elimination not yet supported"
    graph = to_format(graph, 'csr') > 0
    existence = ones(primal(graph), dtype=bool)
    while True:
        count = dsum(graph * existence.Ti)
        enough = (count >= min_edges) | ~mask
        if (enough == existence).all:
            return existence
        existence &= enough


def only_with_min_edge_count(graph: Tensor, data, directed=False, min_edges=2, mask: Tensor = None):
    valid = min_valid_edge_count_mask(graph, directed=directed, min_edges=min_edges, mask=mask)
    lookup = {valid.shape.name: valid, valid.shape.as_dual().name: valid.Ti}
    return math.slice(data, lookup)
    # def keep_valid(x: Tensor):
    #     return x[lookup]
    # return tree_map(keep_valid, data)


def limited_eliminate(graph: Tensor, threshold: float, min_keep=0, directed=False):
    """
    Eliminates graph edges with values below `threshold`, but keeps at least `min_keep` outgoing edges for each node (the highest-value ones if more would be eliminated).

    Args:
        graph:
        threshold:
        min_keep:

    Returns:

    """
    csr = csr_matrix(graph.numpy([primal, dual]))
    if directed:
        result = _directed_limited_eliminate_np(csr, threshold, min_keep)
    else:
        result = _undirected_limited_eliminate_np(csr, threshold, min_keep)
    return wrap(result, primal(graph) + dual(graph))


def _directed_limited_eliminate_np(graph: csr_matrix, threshold: float, min_keep: int):
    data = graph.data
    indices = graph.indices
    indptr = graph.indptr
    keep = data >= threshold  # Initial threshold mask
    row_counts = np.diff(indptr)  # Count retained edges per row
    kept_counts = np.add.reduceat(keep.astype(np.int32), indptr[:-1])
    rows = np.flatnonzero((kept_counts < min_keep) & (row_counts > kept_counts))  # Rows needing supplementation
    for r in rows:
        start, end = indptr[r], indptr[r + 1]
        vals = data[start:end]
        k = min(min_keep, len(vals))
        topk = np.argpartition(vals, -k)[-k:]  # top-k within row
        keep[start:end] = False
        keep[start + topk] = True
    return csr_matrix((data[keep], indices[keep], np.concatenate([[0], np.cumsum(np.add.reduceat(keep.astype(np.int32), indptr[:-1]))])), shape=graph.shape)


def _undirected_limited_eliminate_np(graph: csr_matrix, threshold: float, min_keep: int):
    data = graph.data
    indices = graph.indices
    indptr = graph.indptr
    n = graph.shape[0]

    keep = data >= threshold  # Initial threshold mask

    # Count how many edges each row retains after thresholding
    kept_counts = np.add.reduceat(keep.astype(np.int32), indptr[:-1])
    row_counts = np.diff(indptr)

    # Rows that fall below min_keep after thresholding
    needy_rows = set(np.flatnonzero((kept_counts < min_keep) & (row_counts > kept_counts)))

    # For each edge (r, c), also check if the column node c is needy
    # Build a per-edge mask: is either endpoint needy?
    row_ids = np.repeat(np.arange(n), row_counts)  # row index for each stored edge
    col_ids = indices                               # col index for each stored edge

    needy_arr = np.zeros(n, dtype=bool)
    if needy_rows:
        needy_arr[list(needy_rows)] = True

    # An edge must be kept if either endpoint is needy
    either_needy = needy_arr[row_ids] | needy_arr[col_ids]

    # Force-keep edges where either endpoint needs more edges
    keep |= either_needy

    # Now supplement: for any still-needy row, ensure at least min_keep edges are kept
    kept_counts2 = np.add.reduceat(keep.astype(np.int32), indptr[:-1])
    rows_still_needy = np.flatnonzero((kept_counts2 < min_keep) & (row_counts > kept_counts2))

    for r in rows_still_needy:
        start, end = indptr[r], indptr[r + 1]
        vals = data[start:end]
        k = min(min_keep, len(vals))
        topk = np.argpartition(vals, -k)[-k:]
        keep[start:end] = False
        keep[start + topk] = True

    new_kept = np.add.reduceat(keep.astype(np.int32), indptr[:-1])
    new_indptr = np.concatenate([[0], np.cumsum(new_kept)])
    return csr_matrix(
        (data[keep], indices[keep], new_indptr),
        shape=graph.shape
    )


