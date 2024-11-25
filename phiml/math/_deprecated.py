from typing import Union

from . import Tensor, DimFilter, channel, length, clip, safe_div, tensor
from ._ops import stack_tensors


def clip_length(vec: Tensor, min_len=0, max_len=1, vec_dim: DimFilter = channel, eps: Union[float, Tensor] = None):
    """
    Clips the length of a vector to the interval `[min_len, max_len]` while keeping the direction.
    Zero-vectors remain zero-vectors.

    Args:
        vec: `Tensor`
        min_len: Lower clipping threshold.
        max_len: Upper clipping threshold.
        vec_dim: Dimensions to compute the length over. By default, all channel dimensions are used to compute the vector length.
        eps: Minimum vector length. Use to avoid `inf` gradients for zero-length vectors.

    Returns:
        `Tensor` with same shape as `vec`.
    """
    le = length(vec, vec_dim, eps)
    new_length = clip(le, min_len, max_len)
    return vec * safe_div(new_length, le)


def cross_product(vec1: Tensor, vec2: Tensor) -> Tensor:
    """
    Computes the cross product of two vectors in 2D.

    Args:
        vec1: `Tensor` with a single channel dimension called `'vector'`
        vec2: `Tensor` with a single channel dimension called `'vector'`

    Returns:
        `Tensor`
    """
    vec1 = tensor(vec1)
    vec2 = tensor(vec2)
    spatial_rank = vec1.vector.size if 'vector' in vec1.shape else vec2.vector.size
    if spatial_rank == 2:  # Curl in 2D
        assert vec2.vector.exists
        if vec1.vector.exists:
            v1_x, v1_y = vec1.vector
            v2_x, v2_y = vec2.vector
            return v1_x * v2_y - v1_y * v2_x
        else:
            v2_x, v2_y = vec2.vector
            return vec1 * stack_tensors([-v2_y, v2_x], channel(vec2))
    elif spatial_rank == 3:  # Curl in 3D
        assert vec1.vector.exists and vec2.vector.exists, f"Both vectors must have a 'vector' dimension but got shapes {vec1.shape}, {vec2.shape}"
        v1_x, v1_y, v1_z = vec1.vector
        v2_x, v2_y, v2_z = vec2.vector
        return stack_tensors([
            v1_y * v2_z - v1_z * v2_y,
            v1_z * v2_x - v1_x * v2_z,
            v1_x * v2_y - v1_y * v2_x,
        ], vec1.shape['vector'])
    else:
        raise AssertionError(f'dims = {spatial_rank}. Vector product not available in > 3 dimensions')


