import warnings
from typing import Sequence

import numpy as np
from scipy.sparse.linalg import aslinearoperator

from ..backend import get_precision, choose_backend
from ._shape import Shape, instance, EMPTY_SHAPE, dual, channel, isize, DEBUG_CHECKS, non_batch, primal, batch
from ._magic_ops import pack_dims, expand, concat, rename_dims
from ._tensors import Tensor, wrap, reshaped_tensor
from ._sparse import get_format, to_format, sparse_tensor, is_sparse, dense, native_matrix
from ._lin_trace import LinTracer, dependent_src_dims
from . import _ops as math
from ..backend._dtype import INT64


def min_rank_deficiency(matrix: Tensor) -> Tensor:
    """ A matrix is rank deficient if
    * All rows sum to 0 (because A @ 1 = row-sum)
    * Any row is fully 0
    """
    if matrix._prop.min_rank_deficiency is not None:
        return matrix._prop.min_rank_deficiency
    elif matrix._prop.tracer is not None:
        tracer = matrix._prop.tracer
        if isinstance(tracer, LinTracer):
            if not tracer._fac.available:
                return wrap(isize(guaranteed_empty_rows(matrix)))
            fac = math.convert(tracer._fac, math.NUMPY)
            deficiency = wrap(0)
            row_sums = math.sum_(fac, '_deps')
            abs_row_sums = math.sum_(abs(fac), '_deps')
            eps = {16: 1e-2, 32: 1e-5, 64: 1e-10}[get_precision()]
            is_balanced = math.close(0, row_sums, rel_tolerance=0, abs_tolerance=eps * abs_row_sums, reduce=tracer._indices.shape)
            deficiency += is_balanced
            deficiency += math.sum_(abs_row_sums == 0, tracer._indices.shape)  # number of zero rows
            return deficiency
    else:
        row_sums = math.sum_(matrix, dual)
        abs_row_sums = math.sum_(abs(matrix), dual)
        eps = {16: 1e-2, 32: 1e-5, 64: 1e-10}[get_precision()]
        is_balanced = math.close(0, math.stop_gradient(row_sums), rel_tolerance=0, abs_tolerance=eps * abs_row_sums, reduce=non_batch)
        return math.where(is_balanced, 1, 0)


def guaranteed_empty_rows(matrix: Tensor) -> Tensor:
    if matrix._prop.empty_rows is not None:
        return matrix._prop.empty_rows
    elif matrix._prop.tracer is not None:
        tracer = matrix._prop.tracer
        if isinstance(tracer, LinTracer):
            fac_nz = tracer._fac != 0 if tracer._fac.available else tracer._fac_nz
            is_zero_row = ~math.any_(fac_nz, '_deps')
            return math.nonzero(is_zero_row, list_dim=instance('empties'), element_dims=None, list_dims=is_zero_row.shape)
    else:
        if matrix.available:
            is_zero_row = math.sum_(abs(matrix), dual) == 0
            return math.nonzero(is_zero_row, list_dim=instance('empties'), element_dims=None, list_dims=is_zero_row.shape)
        else:
            return math.zeros(instance(empties=0), channel(vector=primal(matrix).name_list))


def guaranteed_empty_cols(matrix: Tensor) -> Tensor:
    if matrix._prop.empty_cols is not None:
        return matrix._prop.empty_cols
    elif matrix._prop.tracer is not None:
        src_names = dual(matrix).as_batch().names
        tracer = matrix._prop.tracer
        if isinstance(tracer, LinTracer):
            src_dims = tracer._source.shape[src_names]
            indices = tracer._source_indices(src_dims)
            entry_dims = indices.shape - 'idx'
            entry_dim = instance(entries=entry_dims.volume)
            flat_indices = pack_dims(indices, entry_dims, entry_dim)
            flat_nz = pack_dims(expand(tracer._fac_nz, entry_dims), entry_dims, entry_dim)
            nz_indices = flat_indices[flat_nz]
            src_unused = math.scatter(expand(True, dependent_src_dims(tracer)), nz_indices, False)
            return math.nonzero(src_unused, list_dim=instance('empties'), element_dims=None, list_dims=src_unused.shape)
    else:
        if matrix.available:
            is_zero_col = math.sum_(abs(matrix), primal) == 0
            return math.nonzero(is_zero_col, list_dim=instance('empties'), element_dims=None, list_dims=is_zero_col.shape)
        else:
            return math.zeros(instance(empties=0), channel(vector=primal(matrix).name_list))


# def zero_rows(matrix: Tensor) -> Tensor[int]:
#     if matrix._is_tracer:
#         if not matrix._fac.available:  # use _fac_nz
#             return guaranteed_empty_rows(matrix)
#         else:  # we can use actual entries
#             abs_row_sums = math.sum_(abs(matrix._fac), '_deps')
#     else:
#         abs_row_sums = math.sum_(abs(matrix), dual)
#     is_zero_row = abs_row_sums == 0
#     return math.nonzero(is_zero_row, element_dims=None, list_dims=is_zero_row.shape)
#
#
# def zero_columns(matrix: Tensor) -> Tensor[int]:
#     abs_col_sum = math.sum_(abs(matrix), primal)
#     is_zero_col = abs_col_sum == 0
#     return math.nonzero(is_zero_col, element_dims=None, list_dims=is_zero_col.shape)


def matrix_rank(matrix: Tensor) -> Tensor:
    """
    Approximates the rank of a matrix.
    The tolerances used depend on the current precision.

    Args:
        matrix: Sparse or dense matrix, i.e. `Tensor` with primal and dual dims.

    Returns:
        Matrix rank.
    """
    if is_sparse(matrix):
        # stored_rank = matrix._matrix_rank
        # if (stored_rank >= 0).all:
        #     return stored_rank
        warnings.warn("Matrix rank for sparse matrices is experimental and may not be accurate for large matrices.")
        from scipy.linalg.interpolative import estimate_rank
        eps = {16: 1e-2, 32: 1e-5, 64: 1e-10}[get_precision()]
        def single_sparse_rank(matrix: Tensor) -> Tensor:
            def scipy_determine_rank(scipy_matrix):
                if min(scipy_matrix.shape) <= 100:
                    rank = np.linalg.matrix_rank(scipy_matrix.todense())
                    return np.array(rank, dtype=np.int64)
                if scipy_matrix.dtype not in (np.float64, np.complex128):
                    scipy_matrix = scipy_matrix.astype(np.complex128 if scipy_matrix.dtype.kind == 'c' else np.float64)
                rank = estimate_rank(aslinearoperator(scipy_matrix), eps)
                return np.array(rank, dtype=np.int64)
            nat_mat = native_matrix(matrix, matrix.default_backend)
            scipy_result = matrix.default_backend.numpy_call(scipy_determine_rank, (), INT64, nat_mat)
            return wrap(scipy_result)
        from ._ops import broadcast_op
        return broadcast_op(single_sparse_rank, [matrix], batch(matrix))
    else:  # dense
        native = matrix.native([batch, primal, dual], force_expand=True)
        ranks_native = choose_backend(native).matrix_rank_dense(native)
        return reshaped_tensor(ranks_native, [batch(matrix)], convert=False)


def drop_rows_and_cols_from_system(matrix: Tensor, primal_x_shape: Shape, empty_rows: Tensor, empty_cols: Tensor):
    original_format = get_format(matrix)
    rows = matrix.shape.primal
    cols = matrix.shape.dual
    primal_cols = primal_x_shape[cols.as_channel().name_list]
    if original_format in {'csr', 'csc'}:
        matrix = matrix.decompress()  # no cost if was compressed from coo
    empty_rows = math.ravel_index(empty_rows[rows.name_list], matrix.shape)
    empty_cols = math.ravel_index(empty_cols[primal_cols.name_list], matrix.shape)
    matrix = pack_dims(matrix, rows, channel('rows'))  # respects order given by rows
    matrix = pack_dims(matrix, cols, dual('cols'))
    if DEBUG_CHECKS:
        to_format(matrix, 'csr')  # check no out-of-bounds indices
        assert isize(matrix.rows[empty_rows]._values) == 0  # no values in empty rows
        assert isize(matrix[{dual: empty_cols}]._values) == 0  # no values in empty cols
    # --- Remap matrix indices ---
    get_old_row, get_new_row = index_map_remove(matrix.shape['rows'], empty_rows)
    get_old_col, get_new_col = index_map_remove(matrix.shape['~cols'], empty_cols)
    new_row = get_new_row[matrix.primal_indices()]
    new_col = get_new_col[matrix.dual_indices()]
    if DEBUG_CHECKS:
        assert new_col.min >= 0  # if get_new_row is incorrect, we would get -1 entries
        assert new_row.min >= 0
    new_indices = concat([new_row, new_col], 'idx')
    # --- Construct new matrix ---
    values = matrix._values  # only indices shifted, values are unchanged
    matrix = sparse_tensor(new_indices, values, matrix.shape - isize(empty_rows), can_contain_double_entries=original_format == 'coo', indices_sorted=original_format != 'coo', format=original_format)
    # if DEBUG_CHECKS:
    #     matrix.print('numpy', float_format='.2f')
    #     matrix = dense(matrix).numpy(['rows', '~cols'])
    #     nullity = matrix.shape[0] - np.linalg.matrix_rank(matrix)
    #     inv = np.linalg.inv(matrix)
    # --- Remap inputs/outputs ---
    get_old_col_primal = rename_dims(get_old_col, dual, channel)
    zeros_x = math.zeros(primal_cols)
    get_old_col_exp = math.unravel_index(get_old_col.idx['~cols'].cols.dual.as_instance(), primal_cols)
    zeros_y = math.zeros(rows)
    get_old_row_exp = math.unravel_index(get_old_row.idx['rows'].rows.as_instance(), rows)

    def contract_x(x: Tensor):
        return math.gather(pack_dims(x, cols.as_batch().names, dual('cols')), get_old_col_primal, pref_index_dim='idx')

    def contract_y(y: Tensor):
        return math.gather(pack_dims(y, rows, channel('rows')), get_old_row, pref_index_dim='idx')

    def expand_x(x: Tensor):
        return math.scatter(zeros_x, get_old_col_exp, x.cols.as_instance())

    def expand_y(y: Tensor):
        return math.scatter(zeros_y, get_old_row_exp, y.rows.as_instance())

    return matrix, contract_x, contract_y, expand_x, expand_y


def index_map_remove(dim: Shape, to_remove: Tensor) -> Tensor:
    indices_np = to_remove.numpy([instance])
    keep = np.setdiff1d(np.arange(dim.size), indices_np)
    reverse = np.full(dim.size, -1, dtype=int)
    reverse[keep] = np.arange(keep.size)
    keep = expand(wrap(keep, dim.without_sizes()), channel(idx=dim.name))
    reverse = expand(wrap(reverse, dim.without_sizes()), channel(idx=dim.name))
    return keep, reverse
