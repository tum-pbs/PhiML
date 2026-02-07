from typing import Optional

from ..backend import get_precision
from ._shape import Shape, instance, EMPTY_SHAPE, dual
from ._magic_ops import pack_dims, expand
from ._tensors import Tensor, wrap
from ._lin_trace import LinTracer, dependent_src_dims
from . import _ops as math


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
                return guaranteed_empty_rows(tracer)
            deficiency = wrap(0)
            row_sums = math.sum_(tracer._fac, '_deps')
            abs_row_sums = math.sum_(abs(tracer._fac), '_deps')
            eps = {16: 1e-2, 32: 1e-5, 64: 1e-10}[get_precision()]
            if math.close(0, row_sums, rel_tolerance=0, abs_tolerance=eps * abs_row_sums, reduce=tracer._indices.shape):
                deficiency += 1
            deficiency += math.sum_(abs_row_sums == 0, tracer._indices.shape)  # number of zero rows
            return deficiency
    raise NotImplementedError(matrix)


def guaranteed_empty_rows(matrix: Tensor) -> Tensor:
    if matrix._prop.empty_rows is not None:
        return matrix._prop.empty_rows
    elif matrix._prop.tracer is not None:
        tracer = matrix._prop.tracer
        if isinstance(tracer, LinTracer):
            fac_nz = tracer._fac != 0 if tracer._fac.available else tracer._fac_nz
            is_zero_row = ~math.any_(fac_nz, '_deps')
            return math.nonzero(is_zero_row, list_dim=instance('empties'), element_dims=None, list_dims=is_zero_row.shape)
    raise NotImplementedError(matrix)


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
    raise NotImplementedError(matrix)


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
