"""N-dimensional operations.

This module is only loaded on demand.
"""
import numpy as np

from . import _ops as math
from . import extrapolation as extrapolation
from ._shape import channel, spatial
from ._tensors import Tensor, wrap


def smooth(grid: Tensor, radius: float | Tensor, falloff='gaussian', extrapolation=extrapolation.ZERO_GRADIENT, dims=spatial):
    """
    Smooths `grid` by convolving it with a normalized, radially symmetric kernel.

    The kernel support is chosen automatically from `radius` and truncated to a finite size.
    For scalar radii, the same support is used along all smoothing dimensions.
    When `radius` has a `vector` dimension, it must match `dims` and allows anisotropic smoothing.
    The kernel size is additionally capped by the spatial extent of `grid` along each smoothed dimension.

    Supported falloff modes:

    * `'gaussian'`: weights proportional to `exp(-0.5 * d^2)` with cutoff `4 * radius`
    * `'laplace'`: weights proportional to `exp(-d)` with cutoff `6 * radius`
    * `'linear'`: compact triangular falloff over the selected support

    Args:
        grid: Input `Tensor` to be smoothed.
        radius: Smoothing radius as scalar or `Tensor` with a `vector` dimension matching `dims`.
            The value must be available at tracing time because it determines the kernel size.
            A radius of `0` leaves `grid` unchanged.
        falloff: Kernel profile. One of `'Gaussian'`, `'Laplace'`, `'Linear'`.
        extrapolation: Boundary handling for values sampled outside `grid` during convolution.
        dims: Dimensions along which smoothing is applied.

    Returns:
        Smoothed `Tensor` with the same shape as `grid`.
    """
    assert falloff in ('gaussian', 'laplace', 'linear')
    grid = wrap(grid)
    dims = grid.shape.only(dims, reorder=True)
    if dims.is_empty:
        return grid
    radius = abs(wrap(radius))
    if 'vector' in radius.shape:
        radius = radius.vector[dims]
    else:
        assert radius.rank == 0, f"radius must be a scalar or have a 'vector' dimension matching {dims.names} but got {radius.shape}"
    assert math.all_available(radius), "smooth() requires radius to be available because the kernel size depends on it."
    cutoff_factor = {'gaussian': 4, 'laplace': 6, 'linear': 1}[falloff]
    max_cutoffs = {dim: max(0, (grid.shape.get_size(dim) - 1) // 2) for dim in dims.names}
    if 'vector' in radius.shape:
        cutoffs = {dim: min(int(np.ceil(radius.vector[dim].item() * cutoff_factor)), max_cutoffs[dim]) for dim in dims.names}
    else:
        cutoff = int(np.ceil(radius.item() * cutoff_factor))
        cutoffs = {dim: min(cutoff, max_cutoffs[dim]) for dim in dims.names}
    coords = math.meshgrid(dims=spatial, **{dim: np.arange(-cutoff, cutoff + 1, dtype=np.float32) for dim, cutoff in cutoffs.items()})
    safe_radius = math.maximum(radius, 1e-12)
    distance = math.sqrt(math.sum_((coords / safe_radius) ** 2, channel))
    if falloff == 'gaussian':
        kernel = math.exp(-0.5 * distance ** 2)
    elif falloff == 'laplace':
        kernel = math.exp(-distance)
    elif falloff == 'linear':
        kernel = math.maximum(0, 1 - math.sqrt(math.sum_((coords / (safe_radius + 1)) ** 2, channel)))
    else:
        raise AssertionError(f"Unsupported falloff: {falloff}")
    kernel /= math.sum_(kernel)
    return math.convolve(grid, kernel, 'same', extrapolation, dims)
