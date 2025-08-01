from unittest import TestCase

import numpy as np

from phiml import math, srange
from phiml.backend import Backend
from phiml.backend._backend import init_installed_backends
from phiml.backend._dtype import FLOAT64, FLOAT32, COMPLEX64, COMPLEX128, INT64, INT32, FLOAT16
from phiml.math import extrapolation, spatial, channel, instance, batch, DType, IncompatibleShapes, NAN, vec, \
    non_spatial, wrap, assert_close, PI, tensor, stack, dual, ones, convolve

BACKENDS = init_installed_backends()


def assert_not_close(*tensors, rel_tolerance, abs_tolerance):
    try:
        assert_close(*tensors, rel_tolerance, abs_tolerance)
        raise Exception(AssertionError('Values are not close'))
    except AssertionError:
        pass


class TestOps(TestCase):

    def test_assert_close(self):
        a = spatial(a=10)
        assert_close(math.zeros(a), math.zeros(a), math.zeros(a), rel_tolerance=0, abs_tolerance=0)
        assert_not_close(math.zeros(a), math.ones(a), rel_tolerance=0, abs_tolerance=0)
        for scale in (1, 0.1, 10):
            assert_close(math.zeros(a), math.ones(a) * scale, rel_tolerance=0, abs_tolerance=scale * 1.001)
            assert_close(math.zeros(a), math.ones(a) * scale, rel_tolerance=1, abs_tolerance=0)
            assert_not_close(math.zeros(a), math.ones(a) * scale, rel_tolerance=0.9, abs_tolerance=0)
            assert_not_close(math.zeros(a), math.ones(a) * scale, rel_tolerance=0, abs_tolerance=0.9 * scale)
        with math.precision(64):
            assert_not_close(math.zeros(a), math.ones(a) * 1e-100, rel_tolerance=0, abs_tolerance=0)
            assert_close(math.zeros(a), math.ones(a) * 1e-100, rel_tolerance=0, abs_tolerance=1e-15)

    def test_equal(self):
        self.assertTrue(math.equal(1, 1))
        self.assertTrue(math.equal(1., wrap(1)))
        self.assertTrue(math.equal(None, None))
        self.assertTrue(math.equal('a', 'a'))
        self.assertFalse(math.equal('a', 'b'))
        with math.precision(64):
            self.assertFalse(math.equal(1., 1.000000001))

    def test_close(self):
        # --- non-uniform ---
        c = math.close(0, stack([(1, 0), (1, 0, 1), (0, 0)], batch('stack')), reduce='vector')
        math.assert_close([False, False, True], c)
        c = math.close(0, stack([(1, 0), (1, 0, 1), (0, 0)], batch('stack')))
        self.assertEqual(False, c)
        # --- batched tolerance ---
        tol = wrap([0, 1], batch('b'))
        c = math.close(0, 0.1, abs_tolerance=tol, reduce=None)
        math.assert_close([False, True], c)
        c = math.close(0, 0.1, abs_tolerance=tol)
        self.assertEqual(False, c)

    def test_always_close(self):
        @math.jit_compile
        def jit(x, y):
            if not math.always_close(x, x):
                return 0
            if not math.always_close(x, wrap(x.native())):
                return 0
            if math.always_close(x, y):
                return 0
            if math.always_close(x, 0):
                return 0
            return 1 + 0 * x
        for b in BACKENDS:
            if b.supports(Backend.jit_compile):
                with b:
                    x = math.tensor(0)
                    y = math.tensor(0)
                    self.assertEqual(1, jit(x, y).native(), msg=b.name)
        self.assertFalse(math.always_close(vec(x=0), vec(x=0, y=1)))

    def test_assert_close_non_uniform(self):
        t = math.stack([math.zeros(spatial(x=4)), math.zeros(spatial(x=3))], channel('stack'))
        math.assert_close(t, t+0)
        self.assertFalse(math.close(t, t+1))

    def test_soft_plus(self):
        for backend in BACKENDS:
            with backend:
                x = math.linspace(-4, 4, spatial(samples=5))
                math.assert_close([0.01815, 0.126928, 0.693147, 2.126928, 4.01815], math.soft_plus(x))

    def test_concat(self):
        c = math.concat([math.zeros(spatial(b=3, a=2)), math.ones(spatial(a=2, b=4))], spatial('b'))
        self.assertEqual(2, c.shape.get_size('a'))
        self.assertEqual(7, c.shape.get_size('b'))
        assert_close(c.b[:3], 0)
        assert_close(c.b[3:], 1)

    def test_concat_missing_batch(self):
        t = math.random_normal(instance(particles=2))
        b = math.expand(t, batch(b=2))
        c = math.concat([t, b], 'particles')
        self.assertEqual(batch(b=2) & instance(particles=4), c.shape)

    def test_stack_missing_batch(self):
        t = math.random_normal(instance(particles=2))
        b = math.expand(t, batch(b=2))
        s = math.stack([t, b], channel(c='t,b'))
        self.assertEqual(set(batch(b=2) & instance(particles=2) & channel(c='t,b')), set(s.shape))

    def test_nonzero(self):
        c = math.concat([math.zeros(spatial(b=3, a=2)), math.ones(spatial(a=2, b=4))], spatial('b'))
        nz = math.nonzero(c)
        self.assertEqual(nz.shape.get_size('nonzero'), 8)
        self.assertEqual(nz.shape.get_size('vector'), 2)

    def test_nonzero_batched(self):
        grid = math.tensor([[(0, 1)], [(0, 0)]], batch('batch'), spatial('x,y'))
        nz = math.nonzero(grid)
        self.assertEqual(('batch', 'nonzero', 'vector'), nz.shape.names)
        self.assertEqual(1, nz.batch[0].shape.get_size('nonzero'))
        self.assertEqual(0, nz.batch[1].shape.get_size('nonzero'))

    def test_maximum(self):
        v = math.ones(spatial(x=4, y=3) & channel(vector=2))
        assert_close(math.maximum(0, v), 1)
        assert_close(math.maximum(0, -v), 0)

    def test_finite_min(self):
        t = math.tensor([0, 1, -1, -math.INF, math.INF, math.NAN])
        assert_close(math.finite_min(t), -1)
        t = math.tensor([-math.INF, math.INF, math.NAN])
        assert_close(math.finite_min(t, default=0), 0)

    def test_finite_max(self):
        t = math.tensor([0, 1, -1, -math.INF, math.INF, math.NAN])
        assert_close(math.finite_max(t), 1)
        t = math.tensor([-math.INF, math.INF, math.NAN])
        assert_close(math.finite_max(t, default=0), 0)

    def test_finite_sum(self):
        t = math.tensor([0, 1, 1, -math.INF, math.INF, math.NAN])
        assert_close(math.finite_sum(t), 2)
        t = math.tensor([-math.INF, math.INF, math.NAN])
        assert_close(math.finite_sum(t), math.NAN)

    def test_finite_mean(self):
        t = math.tensor([0, 1, 1, -math.INF, math.INF, math.NAN])
        assert_close(math.finite_mean(t), 2/3)
        t = math.tensor([-math.INF, math.INF, math.NAN])
        assert_close(math.finite_mean(t), math.NAN)

    def test_argmax_argmin(self):
        for b in BACKENDS:
            with b:
                t = math.tensor([0, 2, 1, -1])
                math.assert_close(1, math.argmax(t, 'vector'))
                math.assert_close(3, math.argmin(t, 'vector'))

    def test_sum_collapsed(self):
        ones = math.ones(spatial(x=40000, y=30000))
        assert_close(40000 * 30000, math.sum(ones))

    def test_prod_collapsed(self):
        ones = math.ones(spatial(x=40000, y=30000))
        assert_close(1, math.prod(ones))

    def test_mean_collapsed(self):
        ones = math.ones(spatial(x=40000, y=30000))
        data = math.stack([ones, ones * 2], spatial('vector'))
        assert_close(1.5, math.mean(data))

    def test_std(self):
        t1 = wrap([0, 1, 0, 1], spatial('x'))
        t2 = wrap([[0, 1], [0, 1]], spatial('x,y'))
        t3 = math.stack([wrap([0, 1, 0], spatial('x')), wrap([1], spatial('x'))], spatial('y'))
        math.assert_close(0.5, t1.std, t2.std, t3.std)

    def test_std_collapsed(self):
        ones = math.ones(spatial(x=4, y=3))  # hi-res disabled because the current implementation caches the tensor, causes out-of-memory
        std = math.std(ones)
        assert_close(0, std)

    def test_std_missing_dim(self):
        x = math.random_normal(spatial(x=2))
        std = math.std(x, 'y')
        math.assert_close(0, std)
        self.assertEqual(x.shape, std.shape)

    def test_sum_by_type(self):
        a = math.ones(spatial(x=3, y=4), batch(b=10), instance(i=2), channel(vector=2))
        assert_close(math.sum(a, spatial), 12)

    def test_sum_bool(self):
        for backend in BACKENDS:
            with backend:
                a = math.tensor([True, False, True, False], spatial('x'))
                assert_close(2, math.sum(a))

    def test_at_max(self):
        for backend in BACKENDS:
            with backend:
                # --- 1D ---
                x = math.range(spatial(x=11))
                math.assert_close(-math.max(x, 'x'), math.at_max([-x, x], x, 'x')[0])
                # --- 2D ---
                x = math.meshgrid(x=3, y=3)
                math.assert_close(-math.max(x, 'x'), math.at_max(-x, x, 'x'))
                math.assert_close(-math.max(x, 'vector'), math.at_max(-x, x, 'vector'))
        # --- matrix ---
        m = wrap([[1, 2], [4, 3]], 'row:c,~col')
        a = wrap([1, 2], '~col')
        math.assert_close([2, 1], math.at_max(a, m, '~col'))

    def test_at_min(self):
        for backend in BACKENDS:
            with backend:
                # --- 1D ---
                x = math.range(spatial(x=11))
                math.assert_close(-math.min(x, 'x'), math.at_min(-x, x, 'x'))
                # --- 2D ---
                x = math.meshgrid(x=3, y=3)
                math.assert_close(-math.min(x, 'x'), math.at_min(-x, x, 'x'))
                math.assert_close(-math.min(x, 'vector'), math.at_min(-x, x, 'vector'))

    def test_unstack(self):
        a = math.random_uniform(batch(b=10), spatial(x=4, y=3), channel(vector=2))
        u = math.unstack(a, 'vector')
        self.assertIsInstance(u, tuple)
        self.assertEqual(len(u), 2)
        assert_close(u, math.unstack(a, channel))
        # Multiple dimensions
        u = math.unstack(a, 'x,y')
        self.assertIsInstance(u, tuple)
        self.assertEqual(len(u), 12)
        assert_close(u, math.unstack(a, spatial))

    def test_grid_sample(self):
        for backend in BACKENDS:
            with backend:
                grid = math.sum(math.meshgrid(x=[1, 2, 3], y=[0, 3]), 'vector')  # 1 2 3 | 4 5 6
                coords = math.tensor([(0, 0), (0.5, 0), (0, 0.5), (-2, -1)], instance('list'), channel('vector'))
                interp = math.grid_sample(grid, coords, extrapolation.ZERO)
                assert_close(interp, [1, 1.5, 2.5, 0], msg=backend.name)

    def test_grid_sample_1d(self):
        grid = math.tensor([0, 1, 2, 3], spatial('x'))
        coords = math.tensor([[0], [1], [0.5]], spatial('x'), channel('vector'))
        sampled = math.grid_sample(grid, coords, None)
        math.print(sampled)
        assert_close(sampled, [0, 1, 0.5])

    def test_grid_sample_backend_equality_2d(self):
        grid_yx = math.random_uniform(spatial(y=10, x=7))
        coords_yx = math.random_uniform(batch(mybatch=10) & spatial(x=3, y=2)) * vec(y=12, x=9)
        grid_xy = math.tensor(grid_yx.native('x,y'), spatial('x,y'))
        coords_xy = coords_yx.vector['x,y']
        for extrap in (extrapolation.ZERO, extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC):
            sampled = []
            for backend in BACKENDS:
                with backend:
                    grid_yx = math.tensor(grid_yx)
                    coords_yx = math.tensor(coords_yx)
                    grid_xy = math.tensor(grid_xy)
                    coords_xy = math.tensor(coords_xy)
                    sampled.append(math.grid_sample(grid_yx, coords_yx, extrap))
                    sampled.append(math.grid_sample(grid_xy, coords_xy, extrap))
                    sampled.append(math.grid_sample(grid_xy, coords_yx, extrap))
            assert_close(*sampled, abs_tolerance=1e-5)

    def test_grid_sample_backend_equality_2d_batched(self):
        grid_yx = math.random_uniform(batch(mybatch=10) & spatial(y=10, x=7))
        coords_yx = math.random_uniform(batch(mybatch=10) & spatial(x=3, y=2)) * vec(y=12, x=9)
        grid_xy = math.tensor(grid_yx.native('mybatch,x,y'), batch('mybatch'), spatial('x,y'))
        coords_xy = coords_yx.vector['x,y']
        for extrap in (extrapolation.ZERO, extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC):
            sampled = []
            for backend in BACKENDS:
                with backend:
                    grid_yx = math.tensor(grid_yx)
                    coords_yx = math.tensor(coords_yx)
                    grid_xy = math.tensor(grid_xy)
                    coords_xy = math.tensor(coords_xy)
                    sampled.append(math.grid_sample(grid_yx, coords_yx, extrap))
                    sampled.append(math.grid_sample(grid_xy, coords_xy, extrap))
                    sampled.append(math.grid_sample(grid_xy, coords_yx, extrap))
            assert_close(*sampled, abs_tolerance=1e-5)

    def test_grid_sample_gradient_1d(self):
        def f(grid, coords):
            sampled = math.grid_sample(grid, coords, extrapolation.ZERO)
            return math.mean(math.l2_loss(sampled)) / 2

        f_grad = math.jacobian(f, (0, 1), get_output=False)

        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    grid = math.tensor([0., 1, 2, 3], spatial('x'))
                    coords = math.tensor([0.5, 1.5], instance('points'))
                    grad_grid, grad_coords = f_grad(grid, coords)
                    assert_close(grad_grid, math.tensor([0.125, 0.5, 0.375, 0], spatial('x')), msg=backend)
                    assert_close(grad_coords, math.tensor([0.25, 0.75], instance('points')), msg=backend)

    def test_grid_sample_gradient_2d(self):
        def f(grid, coords):
            sampled = math.grid_sample(grid, coords, extrapolation.ZERO)
            return math.sum(sampled) / 3

        f_grad = math.jacobian(f, (0, 1), get_output=False)

        grads_grid = []
        grads_coords = []
        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    grid = math.tensor([[1., 2, 3], [1, 2, 3]], spatial('x,y'))
                    coords = math.tensor([(0.5, 0.5), (1, 1.1), (-0.8, -0.5)], instance('points'), channel('vector'))
                    grad_grid, grad_coords = f_grad(grid, coords)
                    grads_grid.append(grad_grid)
                    grads_coords.append(grad_coords)
        assert_close(*grads_grid)
        assert_close(*grads_coords)

    def test_closest_grid_values_1d(self):
        grid = math.tensor([0, 1, 2, 3], spatial('x'))
        coords = math.tensor([[0.1], [1.9], [0.5], [3.1]], spatial('x'), channel('vector'))
        closest = math.closest_grid_values(grid, coords, extrapolation.ZERO)
        assert_close(closest, math.tensor([(0, 1), (1, 2), (0, 1), (3, 0)], spatial('x'), channel('closest_x')))

    def test_join_dimensions(self):
        grid = math.random_normal(batch(batch=10) & spatial(x=4, y=3) & channel(vector=2))
        points = math.pack_dims(grid, grid.shape.spatial, instance('points'))
        self.assertEqual({'batch', 'points', 'vector'}, set(points.shape.names))
        self.assertEqual(grid.shape.volume, points.shape.volume)
        self.assertEqual(grid.shape.non_spatial, points.shape.non_instance)

    def test_split_dimension(self):
        grid = math.random_normal(batch(batch=10) & spatial(x=4, y=3) & channel(vector=2))
        points = math.pack_dims(grid, grid.shape.spatial, instance('points'))
        split = points.points.split(grid.shape.spatial)
        self.assertEqual(grid.shape, split.shape)
        assert_close(grid, split)

    def test_cumulative_sum(self):
        t = math.tensor([(0, 1, 2, 3), (-1, -2, -3, -4)], spatial('y,x'))
        for backend in BACKENDS:
            with backend:
                t_ = math.tensor(t)
                x_ = math.cumulative_sum(t_, 'x')
                assert_close(wrap([(0, 1, 3, 6), (-1, -3, -6, -10)], 'y,x'), x_, msg=backend.name)
                y_ = math.cumulative_sum(t_, t.shape[0])
                assert_close(wrap([(0, 1, 2, 3), (-1, -1, -1, -1)], 'y,x'), y_, msg=backend.name)
        # --- multi-dim ---
        t = wrap([[1, 2], [3, 4]], 'y,x')
        math.assert_close(wrap([[1, 2], [4, 6]], 'y,x'), math.cumulative_sum(t, 'y'))
        math.assert_close(wrap([[0, 0], [1, 2], [4, 6]], 'y,x'), math.cumulative_sum(t, 'y', include_0=True))
        math.assert_close(wrap([[0, 0], [1, 2]], 'y,x'), math.cumulative_sum(t, 'y', include_0=True, include_sum=False))
        math.assert_close(wrap([[1, 2]], 'y,x'), math.cumulative_sum(t, 'y', include_0=False, include_sum=False))
        math.assert_close(wrap([[1, 3], [3, 7]], 'y,x'), math.cumulative_sum(t, 'x'))
        math.assert_close(wrap([[0, 1, 3], [0, 3, 7]], 'y,x'), math.cumulative_sum(t, 'x', include_0=True))
        math.assert_close(wrap([[1, 3], [6, 10]], 'y,x'), math.cumulative_sum(t, 'y,x'))
        math.assert_close(wrap([[1, 6], [4, 10]], 'y,x'), math.cumulative_sum(t, 'x,y'))
        math.assert_close(wrap([[0, 1], [3, 6]], 'y,x'), math.cumulative_sum(t, 'y,x', include_sum=False, include_0=True))

    def test_quantile(self):
        for backend in BACKENDS:
            with backend:
                t = math.tensor([(1, 2, 3, 4), (1, 2, 3, 4), (6, 7, 8, 9)], batch('batch'), instance('list'))
                q = math.quantile(t, 0.5)
                math.assert_close([2.5, 2.5, 7.5], q, msg=backend.name)
                q = math.quantile(t, [0.5, 0.6])
                math.assert_close(wrap([(2.5, 2.5, 7.5), (2.8, 2.8, 7.8)], instance('quantiles'), batch('batch')), q, msg=backend.name)

    def test_median(self):
        t = math.tensor([(1, 2, 3, 10), (0, 1, 3, 10)], batch('batch'), instance('list'))
        assert_close(math.median(t), [2.5, 2])

    def test_fft(self):
        def get_2d_sine(grid_size, L):
            indices = np.array(np.meshgrid(*list(map(range, grid_size))))
            phys_coord = indices.T * L / (grid_size[0])  # between [0, L)
            x, y = phys_coord.T
            d = np.sin(2 * np.pi * x + 1) * np.sin(2 * np.pi * y + 1)
            return d

        sine_field = get_2d_sine((32, 32), L=2)
        fft_ref_tensor = math.wrap(np.fft.fft2(sine_field), spatial('x,y'))
        with math.precision(64):
            for backend in BACKENDS:
                if backend.name != 'jax':  # TODO Jax casts to float32 / complex64 on GitHub Actions
                    with backend:
                        sine_tensor = math.tensor(sine_field, spatial('x,y'))
                        fft_tensor = math.fft(sine_tensor)
                        self.assertEqual(fft_tensor.dtype, COMPLEX128, msg=backend.name)
                        assert_close(fft_ref_tensor, fft_tensor, abs_tolerance=1e-12, msg=backend.name)  # Should usually be more precise. GitHub Actions has larger errors than usual.

    def test_ifft(self):
        dimensions = 'xyz'
        for backend in BACKENDS:
            with backend:
                for d in range(1, len(dimensions) + 1):
                    x = math.random_normal(spatial(**{dim: 6 for dim in dimensions[:d]})) + math.tensor((0, 1), batch('batch'))
                    k = math.fft(x)
                    x_ = math.ifft(k)
                    assert_close(x, x_, abs_tolerance=1e-5, msg=backend.name)

    def test_fft_dims(self):
        for backend in BACKENDS:
            with backend:
                x = math.random_normal(batch(x=8, y=6, z=4))
                k3 = math.fft(x, 'x,y,z')
                k = x
                for dim in 'xyz':
                    k = math.fft(k, dim)
                assert_close(k, k3, abs_tolerance=1e-5, msg=backend.name)

    def test_dot_vector(self):
        for backend in BACKENDS:
            with backend:
                a = math.ones(spatial(a=4))
                b = math.ones(spatial(b=4))
                dot = math.dot(a, 'a', b, 'b')
                self.assertEqual(0, dot.rank, msg=backend.name)
                assert_close(dot, 4, a.a * b.b, msg=backend.name)
                assert_close(math.dot(a, 'a', a, 'a'), 4, msg=backend.name)

    def test_dot_matrix(self):
        for backend in BACKENDS:
            with backend:
                a = math.ones(spatial(x=2, a=4) & batch(batch=10))
                b = math.ones(spatial(y=3, b=4))
                dot = math.dot(a, 'a', b, 'b')
                self.assertEqual(set(spatial(x=2, y=3) & batch(batch=10)), set(dot.shape), msg=backend.name)
                assert_close(dot, 4, msg=backend.name)

    def test_dot_batched_vector(self):
        for backend in BACKENDS:
            with backend:
                a = math.ones(batch(batch=10) & spatial(a=4))
                b = math.ones(batch(batch=10) & spatial(b=4))
                dot = math.dot(a, 'a', b, 'b')
                self.assertEqual(batch(batch=10), dot.shape, msg=backend.name)
                assert_close(dot, 4, a.a * b.b, msg=backend.name)
                dot = math.dot(a, 'a', a, 'a')
                self.assertEqual(batch(batch=10), dot.shape, msg=backend.name)
                assert_close(dot, 4, a.a * a.a, msg=backend.name)
                # more dimensions
                a = math.ones(batch(batch=10) & spatial(a=4, x=2))
                b = math.ones(batch(batch=10) & spatial(y=3, b=4))
                dot = math.dot(a, 'a', b, 'b')
                self.assertEqual(set(spatial(x=2, y=3) & batch(batch=10)), set(dot.shape), msg=backend.name)
                assert_close(dot, 4, msg=backend.name)

    def test_dot_missing_multiply(self):
        w1 = math.random_uniform(channel(neurons=64, input=1), low=-1, high=1)
        x = math.random_uniform(batch(batch=100), low=-2, high=2)
        y = x.neurons * w1.input
        self.assertEqual(batch(batch=100) & channel(neurons=64), y.shape)

    def test_range(self):
        for backend in BACKENDS:
            with backend:
                assert_close(math.range(spatial('x'), 1, 5), [1, 2, 3, 4], msg=backend.name)
                assert_close(math.range(spatial('x'), 1), [0], msg=backend.name)
        # --- non-uniform ---
        math.assert_close([1, 2], math.range(instance(i=3), math.range(spatial('x'), 1, 3)).i.size)

    def test_boolean_mask_1d(self):
        for backend in BACKENDS:
            with backend:
                x = math.range(spatial('range'), 4)
                mask = math.tensor([True, False, True, False], spatial('range'))
                assert_close(math.boolean_mask(x, 'range', mask), [0, 2], msg=backend.name)
                assert_close(x.range[mask], [0, 2], msg=backend.name)

    def test_boolean_mask_batched(self):
        for backend in BACKENDS:
            with backend:
                x = math.expand(math.range(spatial('x'), 4), batch(batch=2)) * math.tensor([1, -1])
                mask = math.tensor([[True, False, True, False], [False, True, False, False]], batch('batch'), spatial('x'))
                selected = math.boolean_mask(x, 'x', mask)
                expected_0 = math.tensor([(0, -0), (2, -2)], spatial('x'), channel('vector'))
                expected_1 = math.tensor([(1, -1)], spatial('x'), channel('vector'))
                assert_close(selected.batch[0], expected_0, msg=backend.name)
                assert_close(selected.batch[1], expected_1, msg=backend.name)
                assert_close(selected, x.x[mask], msg=backend.name)

    def test_boolean_mask_semi_batched(self):
        for backend in BACKENDS:
            with backend:
                x = math.range(spatial('x'), 4)
                mask = math.tensor([[True, False, True, False], [False, True, False, False]], batch('batch'), spatial('x'))
                selected = math.boolean_mask(x, 'x', mask)
                self.assertEqual(3, selected.shape.volume, msg=backend.name)

    def test_boolean_mask_dim_missing(self):
        for backend in BACKENDS:
            with backend:
                x = math.random_uniform(spatial(x=2))
                mask = math.tensor([True, False, True, True], batch('selection'))
                selected = math.boolean_mask(x, 'selection', mask)
                self.assertEqual(set(spatial(x=2) & batch(selection=3)), set(selected.shape), msg=backend.name)

    def test_gather_1d(self):
        for backend in BACKENDS:
            with backend:
                t = math.tensor([1, 2, 3, 4], spatial('x'))
                indices = math.wrap([0, 3, 0], instance('points'))
                math.assert_close([1, 4, 1], t[indices], math.gather(t, indices))
                math.assert_close([1, 4, 1], math.gather((-t, t), indices)[1])
                math.assert_close([-1, -4, -1], math.gather((-t, t), indices)[0])
                self.assertIsInstance(math.gather((-t, t), indices), tuple)

    def test_non_uniform_gather(self):
        data = math.tensor([[0, 1, 0, 2, 0, 1], [-1, 0, 1, 0, 2, 1]], batch('b'), spatial('seq'))
        nb_offset = math.tensor([-1, 1], instance('neighbors'))
        zero_positions = math.nonzero(data.seq[1:-1] == 0) + 1
        valid = math.all(data[zero_positions + nb_offset] > 0, 'neighbors')
        result = zero_positions[valid].nonzero[0]
        math.assert_close([2, 3], result.vector['seq'])

    def test_scatter_1d(self):
        for backend in BACKENDS:
            with backend:
                base = math.ones(spatial(x=4))
                indices = math.wrap([1, 2], instance('points'))
                values = math.wrap([11, 12], instance('points'))
                updated = math.scatter(base, indices, values, mode='update', outside_handling='undefined')
                assert_close(updated, [1, 11, 12, 1])
                updated = math.scatter(base, indices, values, mode='add', outside_handling='undefined')
                assert_close(updated, [1, 12, 13, 1])
                # with vector dim
                indices = math.expand(indices, channel(vector=1))
                updated = math.scatter(base, indices, values, mode='update', outside_handling='undefined')
                assert_close(updated, [1, 11, 12, 1])

    def test_scatter_update_1d_batched(self):
        for backend in BACKENDS:
            with backend:
                # Only base batched
                base = math.zeros(spatial(x=4)) + math.tensor([0, 1])
                indices = math.wrap([1, 2], instance('points'))
                values = math.wrap([11, 12], instance('points'))
                updated = math.scatter(base, indices, values, mode='update', outside_handling='undefined')
                assert_close(updated, math.tensor([(0, 1), (11, 11), (12, 12), (0, 1)], spatial('x'), channel('vector')), msg=backend.name)
                # Only values batched
                base = math.ones(spatial(x=4))
                indices = math.wrap([1, 2], instance('points'))
                values = math.wrap([[11, 12], [-11, -12]], batch('batch'), instance('points'))
                updated = math.scatter(base, indices, values, mode='update', outside_handling='undefined')
                assert_close(updated, math.tensor([[1, 11, 12, 1], [1, -11, -12, 1]], batch('batch'), spatial('x')), msg=backend.name)
                # Only indices batched
                base = math.ones(spatial(x=4))
                indices = math.wrap([[0, 1], [1, 2]], batch('batch'), instance('points'))
                values = math.wrap([11, 12], instance('points'))
                updated = math.scatter(base, indices, values, mode='update', outside_handling='undefined')
                assert_close(updated, math.tensor([[11, 12, 1, 1], [1, 11, 12, 1]], batch('batch'), spatial('x')), msg=backend.name)
                # Everything batched
                base = math.zeros(spatial(x=4)) + math.tensor([0, 1], batch('batch'))
                indices = math.wrap([[0, 1], [1, 2]], batch('batch'), instance('points'))
                values = math.wrap([[11, 12], [-11, -12]], batch('batch'), instance('points'))
                updated = math.scatter(base, indices, values, mode='update', outside_handling='undefined')
                assert_close(updated, math.tensor([[11, 12, 0, 0], [1, -11, -12, 1]], batch('batch'), spatial('x')), msg=backend.name)

    def test_scatter_update_2d(self):
        for backend in BACKENDS:
            with backend:
                base = math.ones(spatial(x=3, y=2))
                indices = math.wrap([(0, 0), (0, 1), (2, 1)], instance('points'), channel('vector'))
                values = math.wrap([11, 12, 13], instance('points'))
                updated = math.scatter(base, indices, values, mode='update', outside_handling='undefined')
                assert_close(updated, math.tensor([[11, 1, 1], [12, 1, 13]], spatial('y,x')))

    def test_scatter_add_2d(self):
        for backend in BACKENDS:
            with backend:
                base = math.ones(spatial(x=3, y=2))
                indices = math.wrap([(0, 0), (0, 0), (0, 1), (2, 1)], instance('points'), channel('vector'))
                values = math.wrap([11, 11, 12, 13], instance('points'))
                updated = math.scatter(base, indices, values, mode='add', outside_handling='undefined')
                assert_close(updated, math.tensor([[23, 1, 1], [13, 1, 14]], spatial('y,x')))

    def test_scatter_2d_clamp(self):
        base = math.ones(spatial(x=3, y=2))
        indices = math.wrap([(-1, 0), (0, 2), (4, 3)], instance('points'), channel('vector'))
        values = math.wrap([11, 12, 13], instance('points'))
        updated = math.scatter(base, indices, values, mode='update', outside_handling='clamp')
        assert_close(updated, math.tensor([[11, 1, 1], [12, 1, 13]], spatial('y,x')))

    def test_scatter_2d_discard(self):
        base = math.ones(spatial(x=3, y=2))
        indices = math.wrap([(-1, 0), (0, 1), (3, 1)], instance('points'), channel('vector'))
        values = math.wrap([11, 12, 13], instance('points'))
        updated = math.scatter(base, indices, values, mode='update', outside_handling='discard')
        assert_close(updated, math.tensor([[1, 1, 1], [12, 1, 1]], spatial('y,x')))

    def test_scatter_single(self):
        base = math.zeros(spatial(x=3, y=2))
        indices = vec(x=1, y=0)
        values = 1
        updated = math.scatter(base, indices, values, outside_handling='discard')
        assert_close(updated, math.tensor([[0, 1, 0], [0, 0, 0]], spatial('y,x')))

    def test_scatter_min_max_1d(self):
        for backend in BACKENDS:
            with backend:
                indices = math.tensor(vec(x=[1, 2, 1]))
                values = wrap([1, 2, 3], instance(indices))
                result = math.scatter(spatial(x=3), indices, values, mode='max')
                math.assert_close([float('-inf'), 3, 2], result)
                result = math.scatter(spatial(x=3), indices, values, mode='min')
                math.assert_close([float('inf'), 1, 2], result)

    def test_scatter_prod_1d(self):
        indices = math.tensor(vec(x=[1, 2, 1]))
        values = wrap([2, 3, 5], instance(indices))
        result = math.scatter(spatial(x=3), indices, values, mode=math.prod)
        math.assert_close([1, 10, 3], result)
        base = wrap([2, 3, 4], spatial('x'))
        result = math.scatter(base, indices, values, mode=math.prod)
        math.assert_close([2, 30, 12], result)

    def test_scatter_any_all_1d(self):
        indices = math.tensor(vec(x=[1, 2, 1]))
        values = wrap([1, 0, 0], instance(indices))
        result = math.scatter(spatial(x=3), indices, values, mode=math.any)
        math.assert_close([False, True, False], result)
        result = math.scatter(spatial(x=3), indices, values, mode=math.all)
        math.assert_close([True, False, False], result)

    def test_histogram_1d(self):
        for backend in BACKENDS:
            with backend:
                data = vec(instance('losses'), 0, .11, .11, .21, .11, .21, .31, .51)
                hist, bin_edges, bin_center = math.histogram(data, instance(loss=10))
                assert_close(hist, [1, 0, 3, 0, 2, 0, 1, 0, 0, 1], msg=backend.name)

    def test_sin(self):
        for backend in BACKENDS:
            with backend:
                assert_close(math.sin(math.zeros(spatial(x=4))), 0, abs_tolerance=1e-6, msg=backend.name)
                assert_close(math.sin(math.tensor(math.PI / 2)), 1, abs_tolerance=1e-6, msg=backend.name)
                assert_close(math.sin(math.tensor(math.PI)), 0, abs_tolerance=1e-6, msg=backend.name)
                assert_close(math.sin(math.tensor(math.PI * 3 / 2)), -1, abs_tolerance=1e-6, msg=backend.name)

    def test_cos(self):
        for backend in BACKENDS:
            with backend:
                assert_close(math.cos(math.zeros(spatial(x=4))), 1, abs_tolerance=1e-6, msg=backend.name)
                assert_close(math.cos(math.tensor(math.PI / 2)), 0, abs_tolerance=1e-6, msg=backend.name)
                assert_close(math.cos(math.tensor(math.PI)), -1, abs_tolerance=1e-6, msg=backend.name)
                assert_close(math.cos(math.tensor(math.PI * 3 / 2)), 0, abs_tolerance=1e-6, msg=backend.name)

    def test_erf(self):
        for backend in BACKENDS:
            with backend:
                assert_close(0, math.erf(math.zeros(spatial(x=4))), abs_tolerance=1e-6, msg=backend.name)
                assert_close(1, math.erf(math.zeros(spatial(x=4)) + 1e5), abs_tolerance=1e-6, msg=backend.name)
                assert_close(-1, math.erf(math.zeros(spatial(x=4)) - 1e5), abs_tolerance=1e-6, msg=backend.name)

    def test_trigonometric_hyperbolic(self):
        for f in [math.sin, math.cos, math.tan, math.sinh, math.cosh, math.tanh,
                  math.arcsin, math.arccos, math.arctan, math.arcsinh, math.arccosh, math.arctanh]:
            results = []
            for backend in BACKENDS:
                with backend:
                    value = math.tensor(0.3421)
                    results.append(f(value))
            assert_close(results, msg=f.__name__)

    def test_arccosh(self):
        results = []
        for backend in BACKENDS:
            with backend:
                value = math.tensor(1.3421)
                results.append(math.arccosh(value))
        assert_close(results)

    def test_arctan(self):
        results = []
        for backend in BACKENDS:
            with backend:
                value = math.tensor(1.3421)
                results.append(math.arctan(value, divide_by=0))
        assert_close(results)

    def test_factorial(self):
        for backend in BACKENDS:
            with backend:
                # --- int32 ---
                value = math.to_int32(math.tensor(4))
                math.assert_close(24, math.factorial(value))
                self.assertEqual(INT32, math.factorial(value).dtype)
                # --- int64 ---
                value = math.to_int64(math.tensor(14))
                math.assert_close(87178291200, math.factorial(value))
                self.assertEqual(INT64, math.factorial(value).dtype)
                # --- gamma ---
                math.assert_close(1.791759469228055, math.log_gamma(math.tensor(4)))
                math.assert_close(24., math.factorial(math.tensor(4.)))
                self.assertEqual(float, math.factorial(math.tensor(4.)).dtype.kind)

    def test_incomplete_gamma(self):
        math.assert_close(0.13533528, math.incomplete_gamma(1, 2, upper=True, regularized=False))
        math.assert_close(0.36787945, math.incomplete_gamma(1, 1, upper=True, regularized=False))
        math.assert_close(1, math.incomplete_gamma(1, 1, upper=False, regularized=True) + math.incomplete_gamma(1, 1, upper=True, regularized=True))

    def test_degrees(self):
        rad = [0, PI/4, PI/2, 3/4*PI, PI, -PI]
        math.assert_close([0, 45, 90, 135, 180, -180], math.radians_to_degrees(rad))
        math.assert_close(rad, math.degrees_to_radians(math.radians_to_degrees(rad)))

    def test_angle(self):
        math.assert_close(math.PI/4, math.angle(1+1j))
        # math.assert_close(45, math.degrees(math.angle(1+1j)))
        # math.assert_close(0, math.degrees(math.angle(1)))

    def test_any(self):
        for backend in BACKENDS:
            with backend:
                assert_close(math.any(math.tensor([[False, True], [False, False]], spatial('y,x')), dim='x'), [True, False])
                assert_close(math.any(math.tensor([[False, True], [False, False]], spatial('y,x')), dim='x,y'), True)
                assert_close(math.any(math.tensor([[False, True], [False, False]], spatial('y,x'))), True)

    def test_all(self):
        for backend in BACKENDS:
            with backend:
                assert_close(math.all(math.tensor([[False, True], [True, True]], spatial('y,x')), dim='x'), [False, True])
                assert_close(math.all(math.tensor([[False, True], [True, True]], spatial('y,x')), dim='x,y'), False)
                assert_close(math.all(math.tensor([[False, True], [True, True]], spatial('y,x'))), False)

    def test_imag(self):
        for backend in BACKENDS:
            with backend:
                assert_close(math.imag(math.ones(spatial(x=4))), 0, msg=backend.name)
                assert_close(math.imag(math.ones(spatial(x=4)) * 1j), 1, msg=backend.name)

    def test_real(self):
        for backend in BACKENDS:
            with backend:
                assert_close(math.real(math.ones(spatial(x=4))), 1, msg=backend.name)
                assert_close(math.real(math.ones(spatial(x=4)) * 1j), 0, msg=backend.name)

    def test_conjugate(self):
        for backend in BACKENDS:
            with backend:
                assert_close(math.conjugate(1 + 1j), 1 - 1j, msg=backend.name)
                assert_close(math.conjugate(1j * math.ones()), -1j, msg=backend.name)

    def test_convolution_1d_scalar(self):
        for backend in BACKENDS:
            with backend:
                x_even = math.tensor([1, 2, 3, 4], spatial('x'))
                x_odd = math.tensor([1, 2, 3, 4, 5], spatial('x'))
                identity_kernel1 = math.ones(spatial(x=1))
                identity_kernel3 = math.tensor([0, 1, 0], spatial('x'))
                shift_kernel2 = math.tensor([0, 1], spatial('x'))
                shift_kernel3 = math.tensor([0, 0, 1], spatial('x'))
                # no padding even
                assert_close(x_even, math.convolve(x_even, identity_kernel1, 'valid'), msg=backend.name)
                assert_close(x_even.x[1:-1], math.convolve(x_even, identity_kernel3, 'valid'), msg=backend.name)
                assert_close(x_even.x[1:], math.convolve(x_even, shift_kernel2, 'valid'), msg=backend.name)
                assert_close(x_even.x[2:], math.convolve(x_even, shift_kernel3, 'valid'), msg=backend.name)
                # no padding odd
                assert_close(x_odd, math.convolve(x_odd, identity_kernel1, 'valid'), msg=backend.name)
                assert_close(x_odd.x[1:-1], math.convolve(x_odd, identity_kernel3, 'valid'), msg=backend.name)
                assert_close(x_odd.x[1:], math.convolve(x_odd, shift_kernel2, 'valid'), msg=backend.name)
                assert_close(x_odd.x[2:], math.convolve(x_odd, shift_kernel3, 'valid'), msg=backend.name)
                # zero-padding even
                assert_close(x_even, math.convolve(x_even, identity_kernel1, 'same', 0), msg=backend.name)
                assert_close(x_even, math.convolve(x_even, identity_kernel3, 'same', 0), msg=backend.name)
                assert_close(x_even, math.convolve(x_even, shift_kernel2, 'same', 0), msg=backend.name)
                assert_close([2, 3, 4, 0], math.convolve(x_even, shift_kernel3, 'same', 0), msg=backend.name)
                # zero-padding odd
                assert_close(x_odd, math.convolve(x_odd, identity_kernel1, 'same', 0), msg=backend.name)
                assert_close(x_odd, math.convolve(x_odd, identity_kernel3, 'same', 0), msg=backend.name)
                assert_close(x_odd, math.convolve(x_odd, shift_kernel2, 'same', 0), msg=backend.name)
                assert_close([2, 3, 4, 5, 0], math.convolve(x_odd, shift_kernel3, 'same', 0), msg=backend.name)
                # periodic padding even
                assert_close(x_even, math.convolve(x_even, identity_kernel1, 'same', extrapolation.PERIODIC), msg=backend.name)
                assert_close(x_even, math.convolve(x_even, identity_kernel3, 'same', extrapolation.PERIODIC), msg=backend.name)
                assert_close([2, 3, 4, 1], math.convolve(x_even, shift_kernel2, 'same', extrapolation.PERIODIC), msg=backend.name)
                assert_close([2, 3, 4, 1], math.convolve(x_even, shift_kernel3, 'same', extrapolation.PERIODIC), msg=backend.name)
                # periodic padding odd
                assert_close(x_odd, math.convolve(x_odd, identity_kernel1, 'same', extrapolation.PERIODIC), msg=backend.name)
                assert_close(x_odd, math.convolve(x_odd, identity_kernel3, 'same', extrapolation.PERIODIC), msg=backend.name)
                assert_close([2, 3, 4, 5, 1], math.convolve(x_odd, shift_kernel2, 'same', extrapolation.PERIODIC), msg=backend.name)
                assert_close([2, 3, 4, 5, 1], math.convolve(x_odd, shift_kernel3, 'same', extrapolation.PERIODIC), msg=backend.name)

    def test_convolution_1d_stride(self):
        for backend in [b for b in BACKENDS if b.name in ['numpy', 'torch']]:
            with backend:
                x_even = math.tensor([1, 2, 3, 4], spatial('x'))
                x_odd = math.tensor([1, 2, 3, 4, 5], spatial('x'))
                identity_kernel1 = math.ones(spatial(x=1))
                identity_kernel2 = math.tensor([0, 1], spatial('x'))
                identity_kernel3 = math.tensor([0, 1, 0], spatial('x'))
                shift_kernel3 = math.tensor([0, 0, 1], spatial('x'))
                # --- strided ---
                # strided no padding even
                # assert_close(x_even.x[::2], math.convolve(x_even, identity_kernel1, strides=2), msg=backend.name)
                # assert_close(x_even.x[1:-1:2], math.convolve(x_even, identity_kernel3, strides=2), msg=backend.name)
                # assert_close(x_even.x[1::2], math.convolve(x_even, identity_kernel2, strides=2), msg=backend.name)
                # assert_close(x_even.x[2::2], math.convolve(x_even, shift_kernel3, strides=2), msg=backend.name)
                # # strided no padding odd
                # assert_close(x_odd.x[::2], math.convolve(x_odd, identity_kernel1, strides=2), msg=backend.name)
                # assert_close(x_odd.x[1:-1:2], math.convolve(x_odd, identity_kernel3, strides=2), msg=backend.name)
                # assert_close(x_odd.x[1::2], math.convolve(x_odd, identity_kernel2, strides=2), msg=backend.name)
                # assert_close(x_odd.x[2::2], math.convolve(x_odd, shift_kernel3, strides=2), msg=backend.name)
                # # strided zero-padding even
                # assert_close(x_even.x[::2], math.convolve(x_even, identity_kernel1, strides=2, extrapolation=extrapolation.ZERO), msg=backend.name)
                # assert_close(x_even.x[::2], math.convolve(x_even, identity_kernel3, strides=2, extrapolation=extrapolation.ZERO), msg=backend.name)
                # assert_close(x_even.x[::2], math.convolve(x_even, identity_kernel2, strides=2, extrapolation=extrapolation.ZERO), msg=backend.name)
                # assert_close([2, 4], math.convolve(x_even, shift_kernel3, strides=2, extrapolation=extrapolation.ZERO), msg=backend.name)
                # # strided zero-padding odd
                # assert_close(x_odd.x[::2], math.convolve(x_odd, identity_kernel1, strides=2, extrapolation=extrapolation.ZERO), msg=backend.name)
                # assert_close(x_odd.x[::2], math.convolve(x_odd, identity_kernel3, strides=2, extrapolation=extrapolation.ZERO), msg=backend.name)
                # assert_close(x_odd.x[::2], math.convolve(x_odd, identity_kernel2, strides=2, extrapolation=extrapolation.ZERO), msg=backend.name)
                # assert_close([2, 4, 0], math.convolve(x_odd, shift_kernel3, strides=2, extrapolation=extrapolation.ZERO), msg=backend.name)
                # # strided periodic padding even
                # assert_close(x_even.x[::2], math.convolve(x_even, identity_kernel1, strides=2, extrapolation=extrapolation.PERIODIC), msg=backend.name)
                # assert_close(x_even.x[::2], math.convolve(x_even, identity_kernel3, strides=2, extrapolation=extrapolation.PERIODIC), msg=backend.name)
                # assert_close(x_even.x[::2], math.convolve(x_even, identity_kernel2, strides=2, extrapolation=extrapolation.PERIODIC), msg=backend.name)
                # assert_close([2, 4], math.convolve(x_even, shift_kernel3, strides=2, extrapolation=extrapolation.PERIODIC), msg=backend.name)
                # # strided periodic padding odd
                # assert_close(x_odd.x[::2], math.convolve(x_odd, identity_kernel1, strides=2, extrapolation=extrapolation.PERIODIC), msg=backend.name)
                # assert_close(x_odd.x[::2], math.convolve(x_odd, identity_kernel3, strides=2, extrapolation=extrapolation.PERIODIC), msg=backend.name)
                # assert_close(x_odd.x[::2], math.convolve(x_odd, identity_kernel2, strides=2, extrapolation=extrapolation.PERIODIC), msg=backend.name)
                # assert_close([2, 4, 1], math.convolve(x_odd, shift_kernel3, strides=2, extrapolation=extrapolation.PERIODIC), msg=backend.name)
                # # full, no padding even
                # assert_close(x_even, math.convolve(x_even, identity_kernel1, full=True, extrapolation=0), msg=backend.name)
                # assert_close(math.pad(x_even, (1, 1)), math.convolve(x_even, identity_kernel3, full=True, extrapolation=0), msg=backend.name)
                # assert_close(math.pad(x_even, (0, 1)), math.convolve(x_even, identity_kernel2, full=True, extrapolation=0), msg=backend.name)
                # assert_close(math.pad(x_even, (0, 2)), math.convolve(x_even, shift_kernel3, full=True, extrapolation=0), msg=backend.name)

                for size, out_size in zip([256, 257, 258, 259], [136, 137, 137, 138]):
                    t = math.convolve(srange(x=size), srange(x=18), 'full', 0, strides=2)
                    math.assert_close([17, 98, 235, 420], t[:4])
                    assert t.shape.size == out_size, f"{t.shape.size} != {out_size}"

                for size, out_size in zip([256, 257, 258, 259], [136, 136, 137, 137]):
                    t = math.convolve(srange(x=size), srange(x=17), 'full', 0, strides=2)
                    math.assert_close([16, 92, 220, 392], t[:4])
                    assert t.shape.size == out_size, f"{t.shape.size} != {out_size}"

                # full, zero-padding even
                # assert_close(x_even, math.convolve(x_even, identity_kernel1, 'full', 0))
                # assert_close(math.pad(x_even, (1, 1)), math.convolve(x_even, identity_kernel3, 'full', 0))
                # assert_close(math.pad(x_even, (0, 1)), math.convolve(x_even, identity_kernel2, 'full', 0))
                # assert_close(math.pad(x_even, (0, 2)), math.convolve(x_even, shift_kernel3, 'full', 0))
                # # full, zero-padding odd
                # assert_close(x_odd, math.convolve(x_odd, identity_kernel1, 0, full=True))
                # assert_close(math.pad(x_odd, (1, 1)), math.convolve(x_odd, identity_kernel3, 0, full=True))
                # assert_close(math.pad(x_odd, (0, 1)), math.convolve(x_odd, identity_kernel2, 0, full=True))
                # assert_close(math.pad(x_odd, (0, 2)), math.convolve(x_odd, shift_kernel3, 0, full=True))
                # # periodic padding even
                # assert_close(x_even, math.convolve(x_even, identity_kernel1, extrapolation.PERIODIC))
                # assert_close(x_even, math.convolve(x_even, identity_kernel3, extrapolation.PERIODIC))
                # assert_close(x_even, math.convolve(x_even, identity_kernel2, extrapolation.PERIODIC))
                # assert_close([2, 3, 4, 1], math.convolve(x_even, shift_kernel3, extrapolation.PERIODIC))
                # # periodic padding odd
                # assert_close(x_odd, math.convolve(x_odd, identity_kernel1, extrapolation.PERIODIC))
                # assert_close(x_odd, math.convolve(x_odd, identity_kernel3, extrapolation.PERIODIC))
                # assert_close(x_odd, math.convolve(x_odd, identity_kernel2, extrapolation.PERIODIC))
                # assert_close([2, 3, 4, 5, 1], math.convolve(x_odd, shift_kernel3, extrapolation.PERIODIC))
                # --- strided ---
                # full, strided zero-padding even
                # assert_close([1, 3], math.convolve(x_even, identity_kernel1, 'full', 0, strides=2))
                # assert_close([1, 3, 0], math.convolve(x_even, identity_kernel3, 'full', 0, strides=2))
                # assert_close([2, 4], math.convolve(x_even, identity_kernel2, 'full', 0, strides=2))
                # assert_close([2, 4, 0], math.convolve(x_even, shift_kernel3, 'full', 0, strides=2))
                # # strided zero-padding odd
                # assert_close([1, 3, 5], math.convolve(x_odd, identity_kernel1, 'full', strides=2, extrapolation=0))
                # assert_close([1, 3, 5], math.convolve(x_odd, identity_kernel3, 'full', strides=2, extrapolation=0))
                # assert_close([2, 4, 0], math.convolve(x_odd, identity_kernel2, 'full', strides=2, extrapolation=0))
                # assert_close([2, 4, 0], math.convolve(x_odd, shift_kernel3, 'full', strides=2, extrapolation=0))
                # # strided periodic padding even
                # assert_close(x_even.x[::2], math.convolve(x_even, identity_kernel1, strides=2, extrapolation=extrapolation.PERIODIC))
                # assert_close(x_even.x[::2], math.convolve(x_even, identity_kernel3, strides=2, extrapolation=extrapolation.PERIODIC))
                # assert_close(x_even.x[::2], math.convolve(x_even, identity_kernel2, strides=2, extrapolation=extrapolation.PERIODIC))
                # assert_close([2, 4], math.convolve(x_even, shift_kernel3, strides=2, extrapolation=extrapolation.PERIODIC))
                # # strided periodic padding odd
                # assert_close(x_odd.x[::2], math.convolve(x_odd, identity_kernel1, strides=2, extrapolation=extrapolation.PERIODIC))
                # assert_close(x_odd.x[::2], math.convolve(x_odd, identity_kernel3, strides=2, extrapolation=extrapolation.PERIODIC))
                # assert_close(x_odd.x[::2], math.convolve(x_odd, identity_kernel2, strides=2, extrapolation=extrapolation.PERIODIC))
                # assert_close([2, 4, 1], math.convolve(x_odd, shift_kernel3, strides=2, extrapolation=extrapolation.PERIODIC))

    def test_transpose_convolution_1d_strided(self):
        for backend in [b for b in BACKENDS if b.name in ['numpy', 'torch']]:
            with backend:
                # --- Same ---
                t = math.convolve(srange(x=100), srange(x=9), 'same', transpose=True)
                assert t.shape.size == 100, f"{t.shape.size}"
                math.assert_close([10, 20, 35, 56], t[:4])
                # --- Full ---
                t = math.convolve(srange(x=100), srange(x=9), 'full', transpose=True)
                assert t.shape.size == 92, f"{t.shape.size}"
                math.assert_close([84, 120, 156, 192], t[:4])
                # --- Stride 2 ---
                for size, out_size in zip([136, 137, 137, 138], [256, 258, 258, 260]):
                    t = math.convolve(srange(x=size), srange(x=18), 'full', strides=2, transpose=True)
                    assert t.shape.size == out_size, f"{t.shape.size} != {out_size}"
                    math.assert_close([168, 204, 240, 285], t[:4])

    def test_convolution_1d_batched(self):
        for backend in BACKENDS:
            with backend:
                # only values batched
                x = tensor([[1, 2, 3], [11, 12, 13]], 'b:b,x') * vec(x=2, y=-1)
                identity_kernel1 = ones(spatial(x=1))
                shift_kernel2 = tensor([0, 1], 'x')
                identity_kernel3 = tensor([0, 1, 0], 'x')
                shift_kernel3 = tensor([0, 0, 1], 'x')
                # no padding
                assert_close(x, convolve(x, identity_kernel1, 'valid'))
                assert_close(convolve(x, shift_kernel2, 'valid'), tensor([[2, 3], [12, 13]], 'b:b,x') * vec(x=2, y=-1))
                assert_close(convolve(x, identity_kernel3, 'valid'), tensor([[2], [12]], 'b:b,x') * vec(x=2, y=-1))
                assert_close(convolve(x, shift_kernel3, 'valid'), tensor([[3], [13]], 'b:b,x') * vec(x=2, y=-1))
                # # zero-padding
                assert_close(convolve(x, identity_kernel1, 'same'), tensor([[1, 2, 3], [11, 12, 13]], 'b:b,x') * vec(x=2, y=-1))
                assert_close(convolve(x, shift_kernel2, 'same'), tensor([[1, 2, 3], [11, 12, 13]], 'b:b,x') * vec(x=2, y=-1))
                assert_close(convolve(x, identity_kernel3, 'same'), tensor([[1, 2, 3], [11, 12, 13]], 'b:b,x') * vec(x=2, y=-1))
                assert_close(convolve(x, shift_kernel3, 'same'), tensor([[2, 3, 0], [12, 13, 0]], 'b:b,x') * vec(x=2, y=-1))
                # # periodic padding
                assert_close(convolve(x, identity_kernel1, 'same', extrapolation.PERIODIC), tensor([[1, 2, 3], [11, 12, 13]], 'b:b,x') * vec(x=2, y=-1))
                assert_close(convolve(x, identity_kernel3, 'same', extrapolation.PERIODIC), tensor([[1, 2, 3], [11, 12, 13]], 'b:b,x') * vec(x=2, y=-1))
                assert_close(convolve(x, shift_kernel2, 'same', extrapolation.PERIODIC), tensor([[2, 3, 1], [12, 13, 11]], 'b:b,x') * vec(x=2, y=-1))
                assert_close(convolve(x, shift_kernel3, 'same', extrapolation.PERIODIC), tensor([[2, 3, 1], [12, 13, 11]], 'b:b,x') * vec(x=2, y=-1))
                # values and filters batched
                mixed_kernel = tensor([[0, 1, 0], [0, 0, 1]], 'b:b,x')
                assert_close(convolve(x, mixed_kernel, 'same'), tensor([[1, 2, 3], [12, 13, 0]], 'b:b,x') * vec(x=2, y=-1))
                assert_close(convolve(x, mixed_kernel, 'same', extrapolation.PERIODIC), tensor([[1, 2, 3], [12, 13, 11]], 'b:b,x') * vec(x=2, y=-1))
                # with output channels
                out_matrix = tensor([[1, 0], [0, 1], [1, 1]], 'out:c,~vector')
                kernel = identity_kernel3 * out_matrix
                expected = tensor([
                    [[2, 4, 6], [22, 24, 26]],
                    [[-1, -2, -3], [-11, -12, -13]],
                    [[1, 2, 3], [11, 12, 13]]], 'out:c,b:b,x')
                assert_close(convolve(x, kernel, 'same'), expected)

    def test_convolution_2d(self):
        for backend in BACKENDS:
            with backend:
                values = math.random_normal(spatial(x=64, y=64))
                kernel = math.random_normal(spatial(x=5, y=5))
                values_conv = math.convolve(values, kernel, 'same', extrapolation.PERIODIC)

    def test_reshaped_native(self):
        a = math.random_uniform(channel(vector=2) & spatial(x=4, y=3))
        nat = math.reshaped_native(a, ['batch', a.shape.spatial, 'vector'], force_expand=False)
        self.assertEqual((1, 12, 2), nat.shape)
        nat = math.reshaped_native(a, [batch(batch=10), a.shape.spatial, channel(vector=2)], force_expand=False)
        self.assertEqual((12, 2), nat.shape[1:])
        nat = math.reshaped_native(a, [batch(batch=10), a.shape.spatial, channel(vector=2)], force_expand=['x'])
        self.assertEqual((12, 2), nat.shape[1:])
        nat = math.reshaped_native(a, [batch(batch=10), a.shape.spatial, channel(vector=2)])
        self.assertEqual((10, 12, 2), nat.shape)
        nat = math.reshaped_native(a, [batch(batch=10), a.shape.spatial, channel(vector=2)], force_expand=['batch'])
        self.assertEqual((10, 12, 2), nat.shape)
        nat = math.reshaped_native(a, [a.shape.spatial, channel(vector=2, v2=2)], force_expand=False)
        self.assertEqual((12, 4), nat.shape)
        try:
            math.reshaped_native(a, [channel(vector=2, v2=2)], force_expand=False)
        except ValueError:
            pass
        nat = math.reshaped_native(a, [spatial, non_spatial], force_expand=False)
        self.assertEqual((12, 2), nat.shape)

    def test_reshaped_native_ellipsis(self):
        a = math.random_uniform(channel(vector=2) & spatial(x=4, y=3))
        nat = math.reshaped_native(a, [batch, ..., 'vector'], force_expand=False)
        self.assertEqual((1, 12, 2), nat.shape)

    def test_native(self):
        nat = np.zeros(4)
        self.assertIs(math.native(nat), nat)
        assert_close(math.native(math.tensor(nat)), nat)

    def test_numpy(self):
        nat = np.zeros(4)
        self.assertIs(math.numpy(nat), nat)
        assert_close(math.numpy(math.tensor(nat)), nat)

    def test_rename_dims(self):
        t = math.zeros(spatial(x=5, y=4))
        self.assertEqual(math.rename_dims(t, 'x', 'z').shape.get_dim_type('z'), 'spatial')
        self.assertEqual(math.rename_dims(t, ['x'], ['z']).shape.get_dim_type('z'), 'spatial')
        self.assertEqual(math.rename_dims(t, ['x'], channel('z')).shape.get_dim_type('z'), 'channel')

    def test_safe_div(self):
        for backend in BACKENDS:
            with backend:
                one = math.ones()
                zero = math.zeros()
                nan = zero / zero
                # inf = one / zero
                assert_close(math.safe_div(zero, one), zero)
                assert_close(math.safe_div(one, zero), zero)
                assert_close(math.safe_div(zero, zero), zero)
                assert_close(math.safe_div(zero, nan), nan)
                assert_close(math.safe_div(nan, one), nan)

    def test_safe_mul(self):
        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    x = tensor([2., 2., NAN], batch('x'))
                    y = tensor([3., NAN, 3.], x.shape)
                    prod, (dx, dy) = math.gradient(lambda x, y: x * y, 'x,y')(x, y)
                    prods, (dxs, dys) = math.gradient(math.safe_mul, 'x,y')(x, y)
                    math.assert_close(prod, prods)
                    math.assert_close(math.where(math.is_finite(dx), dx, 0), dxs)
                    math.assert_close(math.where(math.is_finite(dy), dy, 0), dys)

    def test_random_int(self):
        for backend in BACKENDS:
            with backend:
                # 32 bits
                a = math.random_uniform(instance(values=1000), low=-1, high=1, dtype=(int, 32))
                self.assertEqual(a.dtype, INT32, msg=backend.name)
                self.assertEqual(a.min, -1, msg=backend.name)
                self.assertEqual(a.max, 0, msg=backend.name)
                # 64 bits
                a = math.random_uniform(instance(values=1000), low=-1, high=1, dtype=(int, 64))
                self.assertEqual(a.dtype.kind, int, msg=backend.name)  # Jax may downcast 64-bit to 32
                self.assertEqual(a.min, -1, msg=backend.name)
                self.assertEqual(a.max, 0, msg=backend.name)

    def test_random_complex(self):
        for backend in BACKENDS:
            with backend:
                a = math.random_uniform(instance(values=4), low=-1, high=0, dtype=(complex, 64))
                self.assertEqual(a.dtype, COMPLEX64, msg=backend.name)
                assert_close(a.imag, 0, msg=backend.name)

    def test_random_uniform_batched_limit(self):
        t = math.random_uniform(instance(i=10), low=wrap([0, -1], batch(b=2)))
        self.assertEqual(2, t.rank)

    def test_cast(self):
        for backend in BACKENDS:
            with backend:
                x = math.random_uniform(dtype=FLOAT64)
                self.assertEqual(FLOAT32, math.to_float(x).dtype, msg=backend.name)
                self.assertEqual(COMPLEX64, math.to_complex(x).dtype, msg=backend.name)
                with math.precision(64):
                    self.assertEqual(FLOAT64, math.to_float(x).dtype, msg=backend.name)
                    self.assertEqual(COMPLEX128, math.to_complex(x).dtype, msg=backend.name)
                self.assertEqual(INT64, math.to_int64(x).dtype, msg=backend.name)
                self.assertEqual(INT32, math.to_int32(x).dtype, msg=backend.name)
                self.assertEqual(FLOAT16, math.cast(x, FLOAT16).dtype, msg=backend.name)
                self.assertEqual(COMPLEX128, math.cast(x, COMPLEX128).dtype, msg=backend.name)
                try:
                    math.cast(x, DType(float, 3, False, 1, 2, True, True))
                    self.fail(msg=backend.name)
                except KeyError:
                    pass

    def test_expand_copy_item_names(self):
        a = math.zeros(channel(vector=2))
        try:
            math.expand(a, channel(vector=3))
            self.fail()
        except IncompatibleShapes:
            pass
        b = math.expand(a, channel(vector='x,y'))
        self.assertEqual(('x', 'y'), b.vector.item_names)
        try:
            math.expand(b, channel(vector='a,b'))
            self.fail()
        except IncompatibleShapes:
            pass

    def test_where_nan(self):
        for backend in BACKENDS:
            with backend:
                cond = math.linspace(0, 1, channel(linspace=2)) > 0
                x = math.tensor([-1, -2, -3], spatial('x'))
                t = math.where(cond, x, NAN)
                assert_close(t.linspace[0], NAN)
                assert_close(t.linspace[1], x)

    def test_fit_hyperplane(self):
        for backend in BACKENDS:
            with backend:
                x = math.random_normal(instance(batch=20), channel(vector='x0,x1')) + (1, 1)
                y = math.stack([0.8 * x[0] - x[1] + 1, -0.8 * x[0]], channel(features='y0,y1'))
                from phiml.math._fit import fit_hyperplane
                w, b = fit_hyperplane(x, y, 'batch')
                assert_close(w, wrap([(0.8, -1), (-0.8, 0)], channel(y), channel(x)), abs_tolerance=1e-3)
                assert_close(b, wrap((1, 0), channel(y)), abs_tolerance=1e-3)

    def test_to_device(self):
        for backend in BACKENDS:
            with backend:
                cpu = backend.list_devices('CPU')[0]
                v = math.random_uniform()
                self.assertEqual(cpu, math.to_device(v, 'CPU').device, msg=backend.name)
                self.assertEqual(cpu, math.to_device(v, cpu).device, msg=backend.name)
                for v in [1., backend.random_uniform((), 0, 1, FLOAT32)]:
                    math.to_device(v, 'CPU')
                    math.to_device(v, cpu)

    def test_linspace(self):
        for backend in BACKENDS:
            with backend:
                t = math.linspace(0, 1, instance(x=4))
                math.assert_close([0, 1/3, 2/3, 1], t)
                t = math.linspace(0, 1, spatial(x=4, y=3))
                math.assert_close(math.linspace(0, 1, spatial(x=4)), t.vector['x'])
                math.assert_close(math.linspace(0, 1, spatial(y=3)), t.vector['y'])
                t = math.linspace((0, 1), (1, 2), spatial(x=4, y=3))
                math.assert_close(math.linspace(0, 1, spatial(x=4)), t.vector['x'])
                math.assert_close(math.linspace(1, 2, spatial(y=3)), t.vector['y'])

    def test_slice_off_1d(self):
        t = wrap([1, 2, 3], spatial(x='x1,x2,x3'))
        math.assert_close([1, 2, 3], math.slice_off(t))
        # --- slice off last ---
        math.assert_close([1, 2], math.slice_off(t, {'x': slice(2, None)}))
        math.assert_close([1, 2], math.slice_off(t, {'x': slice(2, 3)}))
        math.assert_close([1, 2], math.slice_off(t, {'x': slice(2, 5)}))
        math.assert_close([1, 2], math.slice_off(t, {'x': slice(-1, None)}))
        math.assert_close([1, 2], math.slice_off(t, {'x': 2}))
        math.assert_close([1, 2], math.slice_off(t, {'x': -1}))
        math.assert_close([1, 2], math.slice_off(t, {'x': 'x3'}))
        # --- slice off first ---
        math.assert_close([2, 3], math.slice_off(t, {'x': 0}))
        math.assert_close([2, 3], math.slice_off(t, {'x': 'x1'}))
        math.assert_close([2, 3], math.slice_off(t, {'x': slice(None, 1)}))
        math.assert_close([2, 3], math.slice_off(t, {'x': slice(0, 1)}))
        # --- slice off center ---
        math.assert_close([1, 3], math.slice_off(t, {'x': 1}))
        math.assert_close([1, 3], math.slice_off(t, {'x': 'x2'}))
        math.assert_close([1, 3], math.slice_off(t, {'x': slice(1, 2)}))
        # --- multi-slice ---
        math.assert_close([2], math.slice_off(t, {'x': slice(None, 1)}, {'x': slice(-1, None)}))
        math.assert_close([2], math.slice_off(t, {'x': 'x1,x3'}))
        math.assert_close([2], math.slice_off(t, {'x': 'x1'}, {'x': 'x3'}))
        math.assert_close([1], math.slice_off(t, {'x': slice(1, 2)}, {'x': slice(2, 3)}))

    def test_slice_off_2d(self):
        t = wrap([[1, 2, 3], [4, 5, 6]], spatial(y='y1,y2', x='x1,x2,x3'))
        for slices in [
            [{'x': 'x3', 'y': 'y2'}],
            [{'x': 2, 'y': 1}],
            [{'x': [2], 'y': [1, 2, 3]}],
            [{'x': slice(2, None), 'y': slice(1, 2)}],
        ]:
            math.assert_close([1, 2, 3], math.slice_off(t, *slices).y['y1'])
            math.assert_close([4, 5], math.slice_off(t, *slices).y['y2'])

    def test_slice_off_non_uniform(self):
        t = math.stack([vec('x', 1, 2), vec('x', 3, 4), vec('x', 5)], batch('y'))
        math.assert_close(wrap([2, 3, 5], batch('y')), math.slice_off(t, {'y': 0, 'x': 0}, {'y': 1, 'x': 1}))

    def test_pad_slices(self):
        ones = math.ones(spatial(x=2), channel(c='x,y'))
        padded = math.pad(ones, [{'c': 'x', 'x': slice(0, 1)}, {'c': 'y', 'x': slice(-1, None)}])
        math.assert_close([0, 1, 1], padded['x'])
        math.assert_close([1, 1, 0], padded['y'])

    def test_meshgrid(self):
        m = math.meshgrid(x=wrap(2))
        math.assert_close([[0], [1]], m)

    def test_transpose(self):
        t = wrap([[0, 1, 2], [3, 4, 5]], spatial('y,x'))
        math.assert_close(t, math.swap_axes(t, 'y,x'))
        math.assert_close(t, math.swap_axes(t, 'x,y'))
        math.assert_close(t, math.swap_axes(t, [1, 0]))
        math.assert_close([[0, 3], [1, 4], [2, 5]], math.swap_axes(t, 'x,y').native('x,y'))
        self.assertEqual(('x', 'y'), math.swap_axes(t, 'x,y').shape.names)

    def test_transpose_native(self):
        t = wrap([[0, 1, 2], [3, 4, 5]], spatial('y,x')).native('y,x')
        math.assert_close([[0, 3], [1, 4], [2, 5]], math.swap_axes(t, [1, 0]))

    def test_sort(self):
        for b in BACKENDS:
            with b:
                t = tensor([[3, 2, 1], [-1, 5, 2]], batch('b'), spatial('x'))
                math.assert_close([[1, 2, 3], [-1, 2, 5]], math.sort(t))
                math.assert_close([[-1, 2, 1], [3, 5, 2]], math.sort(t, 'b'))
                t = tensor([[3, 2, 1], [-1, 5, 2]], batch('b'), spatial(x='1,2,3'))
                self.assertIsNone(math.sort(t).shape.get_item_names('x'))

    def test_sort_by_key(self):
        for b in BACKENDS:
            with b:
                x = tensor([1, 3, 2, -1], spatial('x'))
                result, perm = math.sort((x, range), key=x)
                math.assert_close([-1, 1, 2, 3], result)
                math.assert_close([3, 0, 2, 1], perm)

    def test_pairwise_differences(self):
        for b in BACKENDS:
            with b:
                for format in ['sparse', 'coo', 'csr', 'csc']:
                    for method in ['sparse', 'scipy-kd']:
                        def diffs(points):
                            return math.pairwise_differences(points, max_distance=1.1, format=format, method=method)
                        idx = math.stored_indices(diffs(tensor(vec(x=[0, 100, 900], y=[0, 0, 1]))))
                        math.assert_close([(0, 0), (1, 1), (2, 2)], idx)
                        idx = math.stored_indices(diffs(tensor(vec(x=[0, 1, 900], y=[0, 0, 1]))))
                        math.assert_close([(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)], idx)

    def test_pairwise_differences_jit(self):
        for b in BACKENDS:
            with b:
                for format in ['sparse', 'coo']:  # compressing to 'csr', 'csc' not yet supported in JIT mode
                    for method in ['sparse', 'scipy-kd']:
                        @math.jit_compile
                        def diffs(points):
                            return math.pairwise_differences(points, max_distance=1.1, format=format, method=method)
                        idx = math.stored_indices(diffs(tensor(vec(x=[0, 100, 900], y=[0, 0, 1]))))
                        math.assert_close([(0, 0), (1, 1), (2, 2)], idx.entries[:3])
                        idx = math.stored_indices(diffs(tensor(vec(x=[0, 1, 900], y=[0, 0, 1]))))
                        count = math.count_occurrences(idx, wrap([(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)], 'expected:i', channel(idx)))
                        math.assert_close(1, count)

    def test_pairwise_differences_deltas(self):
        for format in ['dense', 'sparse', 'coo', 'csr', 'csc']:
            for method in ['sparse', 'scipy-kd']:
                def diffs(points):
                    return math.pairwise_differences(points, max_distance=1.1, format=format, method=method)
                result = diffs(tensor(vec(x=[0, 1, 900], y=[0, 0, 1])))
                math.assert_close(vec(x=0, y=0), result.sequence[0:1].sequence.dual[0:1])
                math.assert_close(vec(x=1, y=0), result.sequence[0:1].sequence.dual[1:2])
                math.assert_close(vec(x=-1, y=0), result.sequence[1:2].sequence.dual[0:1])
                math.assert_close(vec(x=0, y=0), result.sequence[1:2].sequence.dual[1:2])

    def test_pairwise_differences_periodic(self):
        for format in ['dense', 'sparse', 'coo', 'csr', 'csc']:
            for method in ['sparse', 'scipy-kd']:
                points = vec(x=(0, 0.99))
                diff = math.pairwise_differences(points, 0.1, format=format, domain=(0, 1), periodic=True, method=method)
                math.assert_close([[0, 0.01], [0.01, 0]], math.norm(diff))

    def test_svd(self):
        for b in BACKENDS:
            with b:
                mat = math.randn(channel(c=3), dual(d=2))
                lat, s, conv = math.svd(mat)
                math.assert_close(mat, conv @ (lat * s), rel_tolerance=1e-5, abs_tolerance=1e-5)
        # --- non-uniform list ---
        matrix = stack([
            math.random_normal(instance(voxels=3), spatial(x=2, y=2)),
            math.random_normal(instance(voxels=5), spatial(x=2, y=2)),
        ], batch('example'))
        latents, s, v = math.svd(matrix, 'x,y', 'example,voxels', 'singular')
        self.assertIn('example', latents.shape)
        self.assertIn('voxels', latents.shape)
        self.assertIn('singular', latents.shape)
        self.assertIn('singular', s.shape)
        self.assertIn('~singular', v.shape)
        self.assertIn('x', v.shape)
        self.assertIn('y', v.shape)

    def test_contains(self):
        a = wrap([(0, 0), (1, 1), (1, 2), (1, 2)], 'points:i,(x,y)')
        b = wrap([(1, 2), (1, 0)], 'query:s,(x,y)')
        math.assert_close([2, 0], math.count_occurrences(a, b))
        math.assert_close([True, False], math.contains(a, b))
