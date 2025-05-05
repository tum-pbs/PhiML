""" Functions related to tensor permutation.
"""
from itertools import permutations
from typing import Union, Optional, Any

import numpy as np

from ..backend import default_backend
from ._shape import concat_shapes_, batch, DimFilter, Shape, SHAPE_TYPES, shape, non_batch, channel, dual, primal, EMPTY_SHAPE
from ._magic_ops import unpack_dim, expand, stack, slice_, squeeze
from ._tensors import reshaped_tensor, TensorOrTree, Tensor, wrap
from ._ops import unravel_index, psum, dmin


def random_permutation(*shape: Union[Shape, Any], dims=non_batch, index_dim=channel('index')) -> Tensor:
    """
    Generate random permutations of the integers between 0 and the size of `shape`.

    When multiple dims are given, the permutation is randomized across all of them and tensor of multi-indices is returned.

    Batch dims result in batches of permutations.

    Args:
        *shape: `Shape` of the result tensor, including `dims` and batches.
        *dims: Sequence dims for an individual permutation. The total `Shape.volume` defines the maximum integer.
            All other dims from `shape` are treated as batch.

    Returns:
        `Tensor`
    """
    assert dims is not batch, f"dims cannot include all batch dims because that violates the batch principle. Specify batch dims by name instead."
    shape = concat_shapes_(*shape)
    assert not shape.dual_rank, f"random_permutation does not support dual dims but got {shape}"
    perm_dims = shape.only(dims)
    batches = shape - perm_dims
    nu = perm_dims.non_uniform_shape
    batches -= nu
    assert nu in shape, f"Non-uniform permutation dims {perm_dims} must be included in the shape but got {shape}"
    b = default_backend()
    result = []
    for idx in nu.meshgrid():
        perm_dims_i = perm_dims.after_gather(idx)
        native = b.random_permutations(batches.volume, perm_dims_i.volume)
        if perm_dims_i.rank == 0:  # cannot add index_dim
            result.append(reshaped_tensor(native, [batches, ()], convert=False))
        else:
            native = b.unravel_index(native, perm_dims_i.sizes)
            result.append(reshaped_tensor(native, [batches, perm_dims_i, index_dim.with_size(perm_dims_i.name_list)], convert=False))
    return stack(result, nu)


def pick_random(value: TensorOrTree, dim: DimFilter, count: Union[int, Shape, None] = 1, weight: Optional[Tensor] = None, same_selection_dims: DimFilter = non_batch, selections: Shape = EMPTY_SHAPE) -> TensorOrTree:
    """
    Pick one or multiple random entries from `value`.

    Args:
        value: Tensor or tree. When containing multiple tensors, the corresponding entries are picked on all tensors that have `dim`.
            You can pass `range` (the type) to retrieve the picked indices.
        dim: Dimension along which to pick random entries. `Shape` with one dim.
        count: Number of entries to pick. When specified as a `Shape`, lists picked values along `count` instead of `dim`.
        weight: Probability weight of each item along `dim`. Will be normalized to sum to 1.
        same_selection_dims: Dims along which to use the same random selection for each element. All other dims except `dim` are treated as batch.
        selections: Additional dims to generate more random subsets. These will be part of the output.

    Returns:
        `Tensor` or tree equal to `value`.
    """
    v_shape = shape(value)
    dim = dim if isinstance(dim, Shape) else v_shape.only(dim)
    same_selection_dims = v_shape.only(same_selection_dims) - dim
    batches = selections + v_shape - dim - same_selection_dims
    nu_dims = v_shape.non_uniform_shape
    u_batches = batches - nu_dims
    assert nu_dims in batches, f"Cannot use same random selection across non-uniform dims but got {nu_dims} while batches are {batches}"
    if count is None and dim.well_defined:
        count = dim.size
    n = dim.volume if count is None else (count.volume if isinstance(count, SHAPE_TYPES) else count)
    b = default_backend()
    idx_slices = []
    for nui in nu_dims.meshgrid():
        u_dim = dim.after_gather(nui)
        nat_weight = weight.native([u_dim]) if weight is not None else None
        if u_dim.volume >= n:
            nat_idx = b.random_subsets(u_dim.volume, subset_size=n, subset_count=u_batches.volume, allow_duplicates=False, element_weights=nat_weight)
        elif u_dim.volume > 0:
            nat_idx = b.range(n) % u_dim.volume
        else:
            raise ValueError(f"Cannot pick random from empty tensor {u_dim}")
        idx = reshaped_tensor(nat_idx, [u_batches, count if isinstance(count, SHAPE_TYPES) else u_dim.without_sizes()], convert=False)
        if count == 1:
            idx = squeeze(count, u_dim)
        # idx = ravel_index()
        idx_slices.append(expand(idx, channel(index=u_dim.name)))
    idx = stack(idx_slices, nu_dims)
    return slice_(value, idx)


def all_permutations(dims: Shape, list_dim=dual('perm'), index_dim: Optional[Shape] = channel('index'), convert=False) -> Tensor:
    """
    Returns a `Tensor` containing all possible permutation indices of `dims` along `list_dim`.

    Args:
        dims: Dims along which elements are permuted.
        list_dim: Single dim along which to list the permutations.
        index_dim: Dim listing vector components for multi-dim permutations. Can be `None` if `dims.rank == 1`.
        convert: Whether to convert the permutations to the default backend. If `False`, the result is backed by NumPy.

    Returns:
        Permutations as a single index `Tensor`.
    """
    np_perms = np.asarray(list(permutations(range(dims.volume))))
    perms = reshaped_tensor(np_perms, [list_dim, dims], convert=convert)
    if index_dim is None:
        assert len(dims) == 1, f"For multi-dim permutations, index_dim must be specified."
        return perms
    return unravel_index(perms, dims, index_dim)


def optimal_perm(cost_matrix: Tensor):
    """
    Given a pair-wise cost matrix of two equal-size vectors, finds the optimal permutation to apply to one vector (corresponding to the dual dim of `cost_matrix`) in order to obtain the minimum total cost for a bijective map.

    Args:
        cost_matrix: Pair-wise cost matrix. Must be a square matrix with a dual and primal dim.

    Returns:
        dual_perm: Permutation that, when applied along the vector corresponding to the dual dim in `cost_matrix`, yields the minimum cost.
        cost: Optimal cost vector, listed along primal dim of `cost_matrix`.
    """
    assert dual(cost_matrix) and primal(cost_matrix), f"cost_matrix must have primal and dual dims but got {cost_matrix}"
    perms = all_permutations(primal(cost_matrix), index_dim=None)
    perms = expand(perms, channel(index=dual(cost_matrix).name_list))
    cost_pairs_by_perm = cost_matrix[perms]
    perm, cost = dmin((perms, cost_pairs_by_perm), key=psum(cost_pairs_by_perm))
    return perm, cost
