"""PhiML.

Project homepage: https://github.com/tum-pbs/PhiML

Documentation overview: https://tum-pbs.github.io/PhiML

PyPI: https://pypi.org/project/phiml/

This package can be used as convenience import, i.e. `from phiml import *`.
"""

import os as _os


with open(_os.path.join(_os.path.dirname(__file__), 'VERSION'), 'r') as version_file:
    __version__ = version_file.read()


# pylint: disable-msg = unused-import
"""
*Main PhiFlow import:* `from phi.flow import *`

Imports important functions and classes from
`math`, `geom`, `field`, `physics` and `vis` (including sub-modules)
as well as the modules and sub-modules themselves.

See `phi.tf.flow`, `phi.torch.flow`, `phi.jax.flow`.
"""

# --- Modules ---
import numpy
import numpy as np
from . import math
from . import backend
from .math import extrapolation
# --- Classes ---
from .math import Shape, Tensor, DType, Solve
# --- Constants ---
from .math import PI, INF, NAN, f
from .math.extrapolation import PERIODIC, ZERO_GRADIENT
# --- Functions ---
from .math import (
    wrap, tensor, vec, zeros, zeros_like, ones, ones_like, linspace, rand, randn, arange, brange, drange, irange, srange, crange, meshgrid,  # Tensor creation
    shape, spatial, channel, batch, instance, dual, primal, dsize, isize, ssize, csize,
    non_spatial, non_channel, non_batch, non_instance, non_dual, non_primal,  # Shape functions (magic)
    unstack, stack, concat, tcat, dcat, icat, scat, ccat, expand, rename_dims, pack_dims, dpack, ipack, spack, cpack, unpack_dim, flatten, cast,  # Magic Ops
    b2i, c2b, c2d, i2b, s2b, si2d, p2d, d2i, d2s, map_s2b, map_i2b, map_c2b, map_d2b, map_d2c, map_c2d,  # dim type conversions
    dsum, psum, isum, ssum, csum, mean, dmean, pmean, imean, smean, cmean, median, sign, round, ceil, floor, sqrt, exp, erf, log, log2, log10, sigmoid, soft_plus,
    dprod, pprod, sprod, iprod, cprod, dmin, pmin, smin, imin, cmin, finite_min, dmax, pmax, smax, imax, cmax, finite_max,
    sin, cos, tan, sinh, cosh, tanh, arcsin, arccos, arctan, arcsinh, arccosh, arctanh, log_gamma, factorial, incomplete_gamma,
    scatter, gather, where, nonzero,
    dot, convolve, maximum, minimum, clip,  # vector math
    safe_div, is_finite, is_nan, is_inf,  # Basic functions
    jit_compile, jit_compile_linear, minimize, gradient as functional_gradient, gradient, solve_linear, solve_nonlinear, iterate, identity,  # jacobian, hessian, custom_gradient # Functional magic
    assert_close, always_close, equal, close,
    l1_loss, l2_loss,
)
# --- Exceptions ---
from .math import ConvergenceException, NotConverged, Diverged



def verify():
    """
    Checks your configuration for potential problems and prints a summary.

    To run verify without importing `phiml`, run the script `tests/verify.py` included in the source distribution.
    """
    import sys
    from ._troubleshoot import assert_minimal_config, troubleshoot
    try:
        assert_minimal_config()
    except AssertionError as fail_err:
        print("\n".join(fail_err.args), file=sys.stderr)
        return
    print(troubleshoot())


def set_logging_level(level='debug'):
    """
    Sets the logging level for Φ-ML functions.

    Args:
        level: Logging level, one of `'critical', 'fatal', 'error', 'warning', 'info', 'debug'`
    """
    from .backend import ML_LOGGER
    ML_LOGGER.setLevel(level.upper())
