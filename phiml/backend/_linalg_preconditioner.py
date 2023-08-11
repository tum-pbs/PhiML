from functools import partial
from typing import Tuple, Callable

import numpy as np
import numpy.linalg

from ._backend import Backend, SolveResult, List, DType, spatial_derivative_evaluation


# def parallelize_dense_triangular_solve(b: Backend, matrix, lower_triangular=True):
#     """
#
#     Args:
#         b:
#         matrix: lower-triangular matrix
#
#     Returns:
#
#     """
#     rows, cols = b.staticshape(matrix)
#     # batch_size, rows, cols, channels = b.staticshape(matrix)
#     xs = {}
#     for row in range(rows):
#         x = b.zeros((cols,))
#         x[row] = 1
#         for j in range(row):
#             x -= xs[j] * matrix[row, j]
#         if not lower_triangular:
#             x /= matrix[row, row]
#         xs[row] = x
#     print(xs)


def solve_chain(b: Backend, matrix_off_diagonal, rhs, diagonal=None, threads=2):
    """

    Args:
        b: `Backend`
        matrix_off_diagonal: (batch, chain-1)
        rhs: (batch_size, chain, channels)
        diagonal: Self-interaction values for each node. (batch, chain_length+1)
            Default is the unit diagonal.
        threads: Amount of parallelization. How many threads to compute in parallel.
            This translates to how many nodes are skipped in the first pass.

    Returns:
        Solution vector.
    """
    assert diagonal is None
    batch_size, chain_len = b.staticshape(rhs)
    assert chain_len >= 2 * threads, f"Chain length must be at least 2*threads but got {chain_len} and threads={threads}"
    assert (chain_len-threads) % threads == 0
    # --- Decompose matrix into (n)-step left, (n-1) step right, all ones? ---
    # --- Solve initial values using a dense solve ---
    start_matrix = b.expand_dims(np.diag(matrix_off_diagonal[0, :threads-1], k=-1), 0)
    x_start = b.solve_triangular_dense(start_matrix, rhs[:, :threads], lower=True, unit_diagonal=diagonal is None)
    # --- Solve next steps in parallel ---
    m_t = b.reshape(matrix_off_diagonal[:, threads-1:], (batch_size, chain_len//threads-1, threads))
    rhs_t = b.reshape(rhs[:, threads:], (batch_size, chain_len//threads-1, threads))
    rhs_t_start = b.concat([b.expand_dims(x_start, 1), rhs_t], 1)
    thread_results = []
    for t in range(threads):
        dense_threads = np.diag(m_t[0, :, t], k=-1)
        dense_threads = b.expand_dims(dense_threads, 0)
        result = b.solve_triangular_dense(dense_threads, rhs_t_start[:, :, t], lower=True, unit_diagonal=diagonal is None)
        thread_results.append(result)
    intermediate_result = np.stack(thread_results, 2)  # (batch_size, chain/threads, threads, channels)
    intermediate_result = b.reshape(intermediate_result, (batch_size, chain_len))
    # --- Second solve consisting of only ones ---
    # The solution is given by: (-1)^n x_n = b0 - b1 + b2 - b3 ... + b_n  except for x_-1 = b_-1
    alternating = (-1) ** np.arange(chain_len-1)[None, :]
    x = b.cumsum(intermediate_result[:, :-1] * alternating, 1)
    x *= alternating
    x = b.concat([x, intermediate_result[:, -1:]], 1)
    x = b.reshape(x, (batch_size, chain_len))
    return x
