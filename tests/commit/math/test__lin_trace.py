from unittest import TestCase

from phiml import math
from phiml.math import expand, spatial, non_dual, extrapolation, vec, wrap, batch, Tensor, arange, dual, channel, concat, rename_dims


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

    def test_matrix_from_function_sparse(self):
        def lin(x):
            l, r = math.shift(x, (0, 1), padding=None, stack_dim=None)
            x = l + r
            y = x[vec(x=[1, 0])]
            y_b = y * wrap([1, -2], batch('b'))
            y = y + y_b
            return y
        test_x = wrap([3, 4, 5], spatial('x'))
        matrix, bias = math.matrix_from_function(lin, test_x)
        math.assert_close(lin(test_x), matrix @ test_x + bias)

    def test_trace_vec_from_scalar(self):
        def lin(x: Tensor):
            return vec(a=2 * x, b=1)
        matrix, bias = math.matrix_from_function(lin, arange(spatial(x='1,2,3')))
        math.assert_close(vec(a=2, b=0), matrix)
        math.assert_close(vec(a=0, b=1), bias)

    def test_trace_simple_affine(self):
        def lin(x: Tensor):
            return 2 * x + 1
        matrix, bias = math.matrix_from_function(lin, arange(spatial(x='1,2,3')))
        math.assert_close(2, matrix)
        math.assert_close(1, bias)

    def test_trace_component_selection(self):
        def lin(x: Tensor):
            return x['y'] - 1 - x['x'] + 0.5
        matrix, bias = math.matrix_from_function(lin, vec(x=1., y=2.))
        math.assert_close(wrap([-1, 1], dual(vector='x,y')), matrix)
        math.assert_close(-0.5, bias)

    def test_trace_vec_from_components(self):
        def lin(x: Tensor):
            return vec('out', a=2*x['y']-x['x'], b=-x['x'], c=-9.81)
        matrix, bias = math.matrix_from_function(lin, vec(x=1., y=2.))
        math.assert_close(wrap([[-1, 2], [-1, 0], [0, 0]], channel(out='a,b,c'), dual(vector='x,y')), matrix)
        math.assert_close(vec('out', a=0, b=0, c=-9.81), bias)

    def test_trace_vec_output(self):
        def lin(x: Tensor):
            return vec(a=2 * x, b=-x)
        matrix, bias = math.matrix_from_function(lin, arange(spatial(x='1,2,3')))
        math.assert_close(vec(a=2, b=-1), matrix)
        math.assert_close(0, bias)

    def test_trace_sum(self):
        def lin(x: Tensor):
            return math.sum(x, 'x')
        matrix, bias = math.matrix_from_function(lin, arange(spatial(x='1,2,3')))
        math.assert_close(wrap([1, 1, 1], dual(x='1,2,3')), matrix)
        math.assert_close(0, bias)

    def test_trace_weighted_sum(self):
        def lin(x: Tensor):
            return math.sum(x * (1, 2, 3) + (4, 5, 6), 'x')
        matrix, bias = math.matrix_from_function(lin, arange(spatial(x='1,2,3')))
        math.assert_close(wrap([1, 2, 3], dual(x='1,2,3')), matrix)
        math.assert_close(15, bias)

    def test_trace_laplace(self):
        def lin(x: Tensor):
            return math.laplace(x, padding='zero-gradient')
        x = arange(spatial(x='1,2,3'))
        matrix, bias = math.matrix_from_function(lin, x)
        math.assert_close(lin(x), matrix @ x + bias)
        math.assert_close(0, bias)

    def test_trace_concat(self):
        def lin(x: Tensor):
            return concat([x.x[:2], -x + 1], 'x')
        x = arange(spatial(x='1,2,3'))
        matrix, bias = math.matrix_from_function(lin, x)
        math.assert_close(lin(x), matrix @ x + bias)
        math.assert_close(wrap([0, 0, 1, 1, 1], spatial(x=5)), bias)

    def test_trace_spatial_gradient(self):
        def lin(x: Tensor):
            return math.spatial_gradient(x, 1., 'forward', 'zero-gradient')
        x = arange(spatial(x='1,2,3'))
        matrix, bias = math.matrix_from_function(lin, x)
        math.assert_close(lin(x), matrix @ x + bias)
        math.assert_close(0, bias)

    def test_trace_double_matmul(self):
        M = wrap([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'x,~x')
        def lin(x: Tensor):
            return M @ (M @ x)
        x = arange(spatial(x='1,2,3'))
        matrix, bias = math.matrix_from_function(lin, x)
        math.assert_close(lin(x), matrix @ x + bias)
        math.assert_close(0, bias)

    def test_trace_convolve_same_periodic(self):
        def lin(x: Tensor):
            return math.convolve(x, wrap([1, -2], 'x'), size='same', extrapolation='periodic')
        x = arange(spatial(x='1,2,3'))
        matrix, bias = math.matrix_from_function(lin, x)
        math.assert_close(lin(x), matrix @ x + bias)
        math.assert_close(0, bias)

    def test_trace_convolve_valid(self):
        def lin(x: Tensor):
            return math.convolve(x, wrap([1, -2], 'x'), size='valid')
        x = arange(spatial(x='1,2,3'))
        matrix, bias = math.matrix_from_function(lin, x)
        math.assert_close(lin(x), matrix @ x + bias)
        math.assert_close(0, bias)

    def test_trace_convolve_full(self):
        def lin(x: Tensor):
            return math.convolve(x, wrap([1, -2], 'x'), size='full')
        x = arange(spatial(x='1,2,3'))
        matrix, bias = math.matrix_from_function(lin, x)
        math.assert_close(lin(x), matrix @ x + bias)
        math.assert_close(0, bias)

    def test_trace_sparse_tensor(self):
        def lin(x: Tensor):
            return math.sparse_tensor(vec(out=wrap([0, 2, 4], 'entries:i')), x.x.as_instance('entries'), spatial(out=6))
        x = arange(spatial(x='1,2,3'))
        matrix, bias = math.matrix_from_function(lin, x)
        math.assert_close(lin(x), matrix @ x + bias)
        math.assert_close(0, bias)

    def test_trace_sparse_sum(self):
        def lin(x: Tensor):
            return math.sum(math.sparse_tensor(vec(out=wrap([0, 2, 4], 'entries:i')), x.x.as_instance('entries'), spatial(out=6)) * (1, 2, 3, 4, 5, 6), 'out')
        x = arange(spatial(x='1,2,3'))
        matrix, bias = math.matrix_from_function(lin, x)
        math.assert_close(lin(x), matrix @ x + bias)
        math.assert_close(0, bias)

    def test_trace_scatter(self):
        def lin(x: Tensor):
            return math.scatter(wrap([1, 2, 3, 4, 5], 'y'), vec(y=wrap([2, 0, 4], 'x:i')), x.x.as_instance())
        x = arange(spatial(x='1,2,3'))
        matrix, bias = math.matrix_from_function(lin, x)
        math.assert_close(lin(x), matrix @ x + bias)

    def test_trace_gather_reorder(self):
        def lin(x: Tensor):
            return (x + (1., 2, 3))[vec(x=[1, 2, 0])].sequence[(1, 2, 0)] - x.x.as_instance('sequence')
        x = arange(spatial(x='1,2,3'))
        matrix, bias = math.matrix_from_function(lin, x)
        math.assert_close(lin(x), matrix @ x + bias)

    def test_trace_rename_dims_add(self):
        def lin(x: Tensor):
            return rename_dims(x, 'vector', 'ren') * .5 + 1 + x
        x = vec(x=1., y=2.)
        matrix, bias = math.matrix_from_function(lin, x)
        math.assert_close(lin(x), matrix @ x + bias)
        math.assert_close(1, bias)

    def test_trace_pack_dims(self):
        def lin(x: Tensor):
            return math.pack_dims(x, 'vector,sequence', 'out:c')
        x = vec(x=[0, 1, 2], y=-1)
        matrix, bias = math.matrix_from_function(lin, x)
        math.assert_close(lin(x), matrix @ x + bias)
        math.assert_close(0, bias)
