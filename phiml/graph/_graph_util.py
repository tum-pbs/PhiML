import numpy as np
from scipy.sparse import csgraph, csr_matrix

from ..math import Tensor, batch, dual, primal, stack, is_sparse, wrap, broadcast, spatial, Shape, channel, expand
from ._scipy_graph import to_scipy_csr


@broadcast(dims=batch)
def ordered_sequences(connectivity_matrix: Tensor, start: Tensor, directed=True, is_endpoint: Tensor = None, sequence_dim: Shape = spatial('sequence')):
    """
    Extract node sequences from the graph described by `connectivity_matrix`.

    Args:
        connectivity_matrix:
        start:
        directed:
        is_endpoint:
        sequence_dim:

    Returns:
        Ordered node index list for every start point.
        For closed loops, first and last index are identical.
    """
    np_matrix = to_scipy_csr(connectivity_matrix)
    nb_count = np_matrix.indptr[1:] - np_matrix.indptr[:-1]
    if is_endpoint is not None:
        is_endpoint = is_endpoint.numpy()
        endpoints = np.flatnonzero(is_endpoint)
        np_matrix[endpoints, :] = 0.
        np_matrix.eliminate_zeros()
    result = []
    for i_start in start:
        i_nb_count = nb_count[i_start]
        assert i_nb_count <= 2, f"Nodes may have at most 2 neighbors to extract ordered sequences, but start node {i_start} has {i_nb_count} neighbors."
        if i_nb_count == 1:
            node_array = csgraph.depth_first_order(np_matrix, int(i_start), directed=directed, return_predecessors=False)
        elif i_nb_count == 2:
            node_array, predecessors = csgraph.depth_first_order(np_matrix, int(i_start), directed=directed, return_predecessors=True)
            branch_starts = np.flatnonzero(predecessors[node_array] == i_start)[1:]
            if len(branch_starts) == 0:  # closed loop
                node_array = np.concatenate([node_array, node_array[:1]])
            else:  # open ends
                assert len(branch_starts) == 1, f"Nodes may have at most 2 neighbors to extract ordered sequences."
                second_branch_idx = branch_starts[0]
                node_array = np.concatenate([node_array[second_branch_idx:][::-1], node_array[:second_branch_idx]])
        else:  # no neighbors
            node_array = np.asarray([i_start])
        result.append(wrap(node_array, sequence_dim))
    return expand(stack(result, start.shape), channel(index=primal(connectivity_matrix)))
