from unittest import TestCase

from phiml import math
from phiml.backend._backend import init_installed_backends
from phiml.math import expand, spatial, non_dual, extrapolation

BACKENDS = init_installed_backends()


class TestTrace(TestCase):

    def test_matrix_from_function(self):
        def simple_gradient(x):
            x0, x1 = math.shift(x, (0, 1), dims='x', padding=extrapolation.ZERO, stack_dim=None)
            return x1 - x0

        def diagonal(x):
            return 2 * x

        for f in [simple_gradient, diagonal]:
            x = expand(1, spatial(x=4))
            matrix, bias = math.matrix_from_function(f, x)
            if math.get_format(matrix) != 'dense':
                matrix = matrix.compress(non_dual)
            math.assert_close(f(x), matrix @ x)
