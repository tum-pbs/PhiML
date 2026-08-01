import numpy as np
from scipy.sparse import csgraph, csr_matrix

from ..math import Tensor, batch, dual, primal, stack, is_sparse, wrap, broadcast, spatial, Shape, channel, expand


def to_scipy_csr(matrix: Tensor):
    assert dual(matrix) and primal(matrix), f"connectivity_matrix must contain dual and primal dims"
    np_matrix = matrix.numpy() if is_sparse(matrix) else csr_matrix(matrix.numpy([primal, dual]))
    np_matrix = np_matrix.tocsr()
    np_matrix.eliminate_zeros()
    return np_matrix


@broadcast(dims=batch)
def connected_components(connectivity_matrix: Tensor, mask: Tensor = None, directed=True, connection='weak'):
    """
    Analyze the connected components of a (sparse) graph.

    Args:
        connectivity_matrix: Square matrix. Must contain one primal and one dual dim. May additionally contain batch dims.

    Returns:
        num_components: Number of connected components
        labels: Component label for each item. Masked elements are marked as -1.
    """
    np_matrix = to_scipy_csr(connectivity_matrix)
    if mask is not None:
        np_mask = mask.numpy()
        valid_idx = np.nonzero(np_mask)[0]
        np_matrix = np_matrix[valid_idx, :][:, valid_idx]
    n, labels = csgraph.connected_components(np_matrix, directed=directed, connection=connection)
    if mask is not None:
        extended_labels = np.full(np_mask.shape, -1, dtype=int)
        extended_labels[valid_idx] = labels
        labels = extended_labels
    return n, wrap(labels, primal(connectivity_matrix))


@broadcast(dims=batch)
def reverse_cuthill_mckee(connectivity_matrix: Tensor, is_symmetric=False):
    """
    Compute the Reverse Cuthill-McKee ordering of a (sparse) graph.
    This orders rows and cols so that entries are concentrated close to the diagonal.

    Args:
        connectivity_matrix: Square matrix. Must contain one primal and one dual dim. May additionally contain batch dims.
        is_symmetric: Whether the matrix guaranteed to be symmetric.

    Returns:
        perm: Permutation of the indices that reduces the bandwidth of the matrix.
    """
    np_matrix = to_scipy_csr(connectivity_matrix)
    perm = csgraph.reverse_cuthill_mckee(np_matrix, symmetric_mode=is_symmetric)
    return wrap(perm, primal(connectivity_matrix))


@broadcast(dims=batch)
def depth_first_order(connectivity_matrix: Tensor, start: Tensor, directed=True):
    np_matrix = to_scipy_csr(connectivity_matrix)
    for i_start in start:
        node_array, predecessors = csgraph.depth_first_order(np_matrix, int(i_start), directed=directed, return_predecessors=False)
    return wrap(node_array, primal(connectivity_matrix))
