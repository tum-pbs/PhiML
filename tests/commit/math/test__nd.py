from itertools import product
from unittest import TestCase
from phiml import math
from phiml.math import wrap, extrapolation, PI, tensor, batch, spatial, instance, channel, NAN, vec

import numpy as np
import os


REF_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'reference_data')


class TestMathNDNumpy(TestCase):

    def test_const_vec(self):
        v = math.const_vec(1, channel(v='x,y'))
        math.assert_close(1, v)
        self.assertEqual(v.shape, channel(v='x,y'))
        v = math.const_vec(1, spatial(x=4, y=3))
        math.assert_close(1, v)
        self.assertEqual(v.shape, channel(vector='x,y'))
        v = math.const_vec(1, 'x,y')
        math.assert_close(1, v)
        self.assertEqual(v.shape, channel(vector='x,y'))

    def test_gradient_scalar(self):
        ones = tensor(np.ones([2, 4, 3]), batch('batch'), spatial('x,y'))
        cases = dict(difference=('central', 'forward', 'backward'),
                     padding=(extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC, extrapolation.SYMMETRIC))
        for case_dict in [dict(zip(cases, v)) for v in product(*cases.values())]:
            scalar_grad = math.spatial_gradient(ones, dx=0.1, **case_dict)
            math.assert_close(scalar_grad, 0)
            self.assertEqual(scalar_grad.shape.names, ('batch', 'x', 'y', 'gradient'))
            ref_shape = (2, 4, 3, 2) if case_dict['padding'] is not None else ((2, 2, 1, 2) if case_dict['difference'] == 'central' else (2, 3, 2, 2))
            self.assertEqual(scalar_grad.shape.sizes, ref_shape)

    def test_gradient_vector(self):
        meshgrid = math.meshgrid(x=4, y=3)
        cases = dict(difference=('central', 'forward', 'backward'),
                     padding=(extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC, extrapolation.SYMMETRIC),
                     dx=(0.1, 1),
                     dims=(spatial, ('x', 'y'), ))
        for case_dict in [dict(zip(cases, v)) for v in product(*cases.values())]:
            grad = math.spatial_gradient(meshgrid, **case_dict)
            inner = grad.x[1:-1].y[1:-1]
            math.assert_close(inner.gradient[0].vector[1], 0)
            math.assert_close(inner.gradient[1].vector[0], 0)
            math.assert_close(inner.gradient[0].vector[0], 1 / case_dict['dx'])
            math.assert_close(inner.gradient[1].vector[1], 1 / case_dict['dx'])
            self.assertEqual(grad.shape.get_size('vector'), 2)
            self.assertEqual(grad.shape.get_size('gradient'), 2)
            ref_shape = (4, 3) if case_dict['padding'] is not None else ((2, 1) if case_dict['difference'] == 'central' else (3, 2))
            self.assertEqual((grad.shape.get_size('x'), grad.shape.get_size('y')), ref_shape)

    def test_gradient_1d_vector(self):
        a = tensor([(0,), (1,), (2,)], spatial('x'), channel('vector'))
        math.assert_close(tensor([0.5, 1, 0.5], spatial('x')), math.spatial_gradient(a))

    def test_vector_laplace(self):
        meshgrid = math.meshgrid(x=(0, 1, 2, 3), y=(0, -1))
        cases = dict(padding=(extrapolation.ZERO, extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC, extrapolation.SYMMETRIC),
                     dx=(0.1, 1),
                     dims=(spatial, ('x',), ('y',), ('x', 'y')))
        for case_dict in [dict(zip(cases, v)) for v in product(*cases.values())]:
            laplace = math.laplace(meshgrid, **case_dict)

    # Fourier Poisson

    def test_downsample2x(self):
        meshgrid = math.meshgrid(x=(0, 1, 2, 3), y=(0, -1, -2))
        half_size = math.downsample2x(meshgrid, extrapolation.BOUNDARY)
        math.print(meshgrid, 'Full size')
        math.print(half_size, 'Half size')
        math.assert_close(half_size.vector[0], wrap([[0.5, 2.5], [0.5, 2.5]], spatial('y,x')))
        math.assert_close(half_size.vector[1], wrap([[-0.5, -0.5], [-2, -2]], spatial('y,x')))

    def test_upsample2x(self):
        meshgrid = math.meshgrid(x=(0, 1, 2, 3), y=(0, -1, -2))
        double_size = math.upsample2x(meshgrid, extrapolation.BOUNDARY)
        same_size = math.downsample2x(double_size)
        math.print(meshgrid, 'Normal size')
        math.print(double_size, 'Double size')
        math.print(same_size, 'Same size')
        math.assert_close(meshgrid.x[1:-1].y[1:-1], same_size.x[1:-1].y[1:-1])

    def test_finite_fill_3x3_sanity(self):
        values = tensor([[NAN, NAN, NAN],
                         [NAN, 1,   NAN],
                         [NAN, NAN, NAN]], spatial('x, y'))
        math.assert_close(math.ones(spatial(x=3, y=3)), math.finite_fill(values, distance=1, diagonal=True))
        math.assert_close(math.ones(spatial(x=3, y=3)), math.finite_fill(values, distance=2, diagonal=False))
        values = tensor([[1, 1, 1], [1, NAN, 1], [1, 1, 1]], spatial('x,y'))
        math.assert_close(math.ones(spatial(x=3, y=3)), math.finite_fill(values, distance=1, diagonal=False))
        math.assert_close(math.ones(spatial(x=3, y=3)), math.finite_fill(values, distance=1, diagonal=True))

    def test_finite_fill_3x3(self):
        values = tensor([[NAN, NAN, NAN],
                         [NAN, NAN, 4  ],
                         [NAN, 2,   NAN]], spatial('x, y'))
        expected_diag = tensor([[NAN, 4,   4],
                                [2,   3,   4],
                                [2,   2,   3]], spatial('x, y'))
        math.assert_close(expected_diag, math.finite_fill(values, distance=1, diagonal=True))
        expected = tensor([[NAN, 3.5, 4],
                           [2.5, 3,   4],
                           [2,   2,   3]], spatial('x, y'))
        math.assert_close(expected, math.finite_fill(values, distance=2, diagonal=False))

    def test_extrapolate_valid_3x3_sanity(self):
        values = tensor([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], spatial('x, y'))
        valid = values
        extrapolated_values, extrapolated_valid = math.masked_fill(values, valid)
        expected_values = math.ones(spatial(x=3, y=3))
        expected_valid = extrapolated_values
        math.assert_close(extrapolated_values, expected_values)
        math.assert_close(extrapolated_valid, expected_valid)

    def test_extrapolate_valid_3x3(self):
        valid = tensor([[0, 0, 0],
                        [0, 0, 1],
                        [1, 0, 0]], spatial('x, y'))
        values = tensor([[1, 0, 2],
                         [0, 0, 4],
                         [2, 0, 0]], spatial('x, y'))
        expected_valid = tensor([[0, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]], spatial('x, y'))
        expected_values = tensor([[1, 4, 4],
                                  [2, 3, 4],
                                  [2, 3, 4]], spatial('x, y'))
        extrapolated_values, extrapolated_valid = math.masked_fill(values, valid)
        math.assert_close(extrapolated_values, expected_values)
        math.assert_close(extrapolated_valid, expected_valid)

    def test_extrapolate_valid_4x4(self):
        valid = tensor([[0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0]], spatial('x, y'))
        values = tensor([[1, 0, 0, 0],
                         [0, 0, 4, 0],
                         [2, 0, 0, 0],
                         [0, 0, 0, 1]], spatial('x, y'))
        expected_valid = tensor([[1, 1, 1, 1],
                                 [1, 1, 1, 1],
                                 [1, 1, 1, 1],
                                 [1, 1, 1, 1]], spatial('x, y'))
        expected_values = tensor([[3, 4, 4, 4],
                                  [2, 3, 4, 4],
                                  [2, 3, 4, 4],
                                  [2, 2, 3.25, 4]], spatial('x, y'))
        extrapolated_values, extrapolated_valid = math.masked_fill(values, valid, 2)
        math.assert_close(extrapolated_values, expected_values)
        math.assert_close(extrapolated_valid, expected_valid)

    def test_extrapolate_valid_3D_3x3x3_1(self):
        valid = tensor([[[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]],
                        [[0, 0, 1],
                         [0, 0, 0],
                         [1, 0, 0]],
                        [[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]]], spatial('x, y, z'))
        values = tensor([[[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]],
                         [[1, 0, 4],
                          [0, 0, 0],
                          [2, 0, 0]],
                         [[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]]], spatial('x, y, z'))
        expected_valid = tensor([[[0, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 0]],
                                 [[0, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 0]],
                                 [[0, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 0]]], spatial('x, y, z'))
        expected_values = tensor([[[0, 4, 4],
                                   [2, 3, 4],
                                   [2, 2, 0]],
                                  [[1, 4, 4],
                                   [2, 3, 4],
                                   [2, 2, 0]],
                                  [[0, 4, 4],
                                   [2, 3, 4],
                                   [2, 2, 0]]], spatial('x, y, z'))
        extrapolated_values, extrapolated_valid = math.masked_fill(values, valid, 1)
        math.assert_close(extrapolated_values, expected_values)
        math.assert_close(extrapolated_valid, expected_valid)

    def test_extrapolate_valid_3D_3x3x3_2(self):
        valid = tensor([[[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]],
                        [[0, 0, 1],
                         [0, 0, 0],
                         [1, 0, 0]],
                        [[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]]], spatial('x, y, z'))
        values = tensor([[[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]],
                         [[1, 0, 4],
                          [0, 0, 0],
                          [2, 0, 0]],
                         [[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]]], spatial('x, y, z'))
        expected_valid = math.ones(spatial(x=3, y=3, z=3))
        expected_values = tensor([[[3, 4, 4],
                                   [2, 3, 4],
                                   [2, 2, 3]],
                                  [[3, 4, 4],
                                   [2, 3, 4],
                                   [2, 2, 3]],
                                  [[3, 4, 4],
                                   [2, 3, 4],
                                   [2, 2, 3]]], spatial('x, y, z'))
        extrapolated_values, extrapolated_valid = math.masked_fill(values, valid, 2)
        math.assert_close(extrapolated_values, expected_values)
        math.assert_close(extrapolated_valid, expected_valid)

    def test_fourier_laplace_2d_periodic(self):
        """test for convergence of the laplace operator"""
        test_params = {
            'size': [16, 32, 40],
            'L': [1, 2, 3],  # NOTE: Cannot test with less than 1 full wavelength
        }
        test_cases = [dict(zip(test_params, v)) for v in product(*test_params.values())]
        for params in test_cases:
            vec = math.meshgrid(x=params['size'], y=params['size'])
            sine_field = math.prod(math.sin(2 * PI * params['L'] * vec / params['size'] + 1), 'vector')
            sin_lap_ref = - 2 * (2 * PI * params['L'] / params['size']) ** 2 * sine_field  # leading 2 from from x-y cross terms
            sin_lap = math.fourier_laplace(sine_field, 1)
            # try:
            math.assert_close(sin_lap, sin_lap_ref, rel_tolerance=0, abs_tolerance=1e-5)
            # except BaseException as e:  # Enable the try/catch to get more info about the deviation
            #     abs_error = math.abs(sin_lap - sin_lap_ref)
            #     max_abs_error = math.max(abs_error)
            #     max_rel_error = math.max(math.abs(abs_error / sin_lap_ref))
            #     variation_str = "\n".join(
            #         [
            #             f"max_absolute_error: {max_abs_error}",
            #             f"max_relative_error: {max_rel_error}",
            #         ]
            #     )
            #     print(f"{variation_str}\n{params}")
            #     raise AssertionError(e, f"{variation_str}\n{params}")

    def test_vector_length(self):
        v = tensor([(0, 0), (1, 1), (-1, 0)], instance('values'), channel('vector'))
        le = math.vec_length(v)
        math.assert_close(le, [0, 1.41421356237, 1])
        le = math.vec_length(v, eps=0.01)
        math.assert_close(le, [1e-1, 1.41421356237, 1])
        le = math.vec_length(wrap(1+1j), eps=0.01)
        math.assert_close(1.41421356237, le)

    def test_vec_normalize(self):
        vec = math.vec_normalize(math.vec(x=1, y=-1, Z=0))
        math.assert_close((0.70710678118, -0.70710678118, 0), vec)

    def test_normalize_to(self):
        vec = math.normalize_to(math.vec(x=1, y=-1, Z=2), 1)
        math.assert_close((.5, -.5, 1), vec)

    def test_cross_product(self):
        # --- 2D ---
        c = math.cross_product(math.vec(x=2, y=0), math.vec(x=0, y=1))
        math.assert_close(2, c)
        # --- 3D ---
        c = math.cross_product(math.vec(x=1, y=0, z=0), math.vec(x=0, y=2, z=0))
        math.assert_close(math.vec(x=0, y=0, z=2), c)
        c = math.cross_product(math.vec(x=0, y=1, z=0), math.vec(x=0, y=2, z=0))
        math.assert_close(math.vec(x=0, y=0, z=0), c)

    def test_rotate_vector(self):
        # --- 2D ---
        vec = math.rotate_vector(math.vec(x=2, y=0), math.PI / 2)
        math.assert_close(math.vec(x=0, y=2), vec, abs_tolerance=1e-5)
        math.assert_close(math.vec(x=2, y=0), math.rotate_vector(vec, math.PI / 2, invert=True), abs_tolerance=1e-5)
        # --- 3D ---
        vec = math.rotate_vector(math.vec(x=2, y=0, z=0), angle=math.vec(x=0, y=math.PI / 2, z=0))
        math.assert_close(math.vec(x=0, y=0, z=-2), vec, abs_tolerance=1e-5)
        math.assert_close(math.vec(x=2, y=0, z=0), math.rotate_vector(vec, angle=math.vec(x=0, y=math.PI / 2, z=0), invert=True), abs_tolerance=1e-5)
        # --- None ---
        math.assert_close(math.vec(x=2, y=0), math.rotate_vector(math.vec(x=2, y=0), None, invert=True))

    def test_dim_mask(self):
        math.assert_close((1, 0, 0), math.dim_mask(spatial('x,y,z'), 'x'))
        math.assert_close((1, 0, 1), math.dim_mask(spatial('x,y,z'), 'x,z'))

    def test_vec_expand(self):
        v = math.vec(x=0, y=math.linspace(0, 1, instance(points=10)))
        self.assertEqual(set(instance(points=10) & channel(vector='x,y')), set(v.shape))

    def test_vec_sequence(self):
        size = vec(batch('size'), 4, 8, 16, 32)
        self.assertEqual(batch(size='4,8,16,32'), size.shape)
        math.assert_close([4, 8, 16, 32], size)
        size = vec(batch('size'), [4, 8, 16, 32])
        self.assertEqual(batch(size='4,8,16,32'), size.shape)
        math.assert_close([4, 8, 16, 32], size)

    def test_vec_component_sequence(self):
        math.assert_close(wrap([(0, 1), (0, 2)], spatial('sequence'), channel(vector='x,y')), vec(x=0, y=(1, 2)))
        math.assert_close(wrap([(0, 1), (0, 2)], instance('sequence'), channel(vector='x,y')), vec(x=0, y=[1, 2]))

    def test_losses(self):
        t = math.wrap([(1, 2), (3, 4), (0, 0)], batch('examples'), spatial('x'))
        math.assert_close((3, 7, 0), math.l1_loss(t))
        math.assert_close((2.5, 12.5, 0), math.l2_loss(t))
        math.assert_close((4.5, 24.5, 0), math.frequency_loss(t, threshold=0))

    def test_abs_square(self):
        math.assert_close(2, math.abs_square(1+1j))

    def test_fourier_poisson(self):
        x = math.random_normal(batch(b=2), spatial(x=20, y=20))
        x -= math.mean(x)
        math.assert_close(x, math.fourier_laplace(math.fourier_poisson(x, .3), .3), abs_tolerance=1e-5)

    def test_sample_subgrid(self):
        grid = math.random_normal(batch(b=2), spatial(x=20, y=20))
        s = math.sample_subgrid(grid, start=vec(x=4, y=.5), size=spatial(x=2, y=3))
        self.assertEqual(batch(grid) & spatial(x=2, y=3), s.shape)

    def test_rotation_matrix(self):
        def assert_matrices_equal(angle):
            matrix = math.rotation_matrix(angle)
            angle_ = math.rotation_angles(matrix)
            math.assert_close(matrix, math.rotation_matrix(angle_), abs_tolerance=1e-5)

        angle = wrap([0, -math.PI/2, math.PI/2, -math.PI, math.PI, 2*math.PI], batch('angles'))
        assert_matrices_equal(angle)
        # --- 3D axis-angle ---
        angle = wrap([0, -math.PI/2, math.PI/2, -math.PI, math.PI, 2*math.PI], batch('angles'))
        assert_matrices_equal(math.vec(x=0, y=0, z=angle))
        assert_matrices_equal(math.vec(x=0, y=angle, z=0))
        assert_matrices_equal(math.vec(x=angle, y=0, z=0))
        assert_matrices_equal(math.vec(x=angle, y=angle, z=0))
        assert_matrices_equal(math.vec(x=angle, y=angle, z=angle))
        # --- 3D Euler angle ---
        angle = wrap([0, -math.PI/2, math.PI/2, -math.PI, math.PI, 2*math.PI], batch('angles'))
        assert_matrices_equal(math.vec('angle', x=0, y=0, z=angle))
        assert_matrices_equal(math.vec('angle', x=0, y=angle, z=0))
        assert_matrices_equal(math.vec('angle', x=angle, y=0, z=0))
        assert_matrices_equal(math.vec('angle', x=angle, y=angle, z=0))
        assert_matrices_equal(math.vec('angle', x=angle, y=angle, z=angle))

    def test_neighbor_reduce(self):
        grid1 = wrap([0, 2, 4, 0], spatial('x'))
        math.assert_close([2, 6, 4], math.neighbor_sum(grid1))
        math.assert_close([1, 3, 2], math.neighbor_mean(grid1))
        math.assert_close([2, 4, 4], math.neighbor_max(grid1))
        math.assert_close([0, 2, 0], math.neighbor_min(grid1))
        grid2 = wrap([[0, 1], [2, 3]], spatial('y,x'))
        math.assert_close([[6]], math.neighbor_sum(grid2))

    def test_at_min_neighbor(self):
        x = math.range(spatial(x=4))
        x_min, neg_x_min = math.at_min_neighbor((x, -x), x, 'x')
        math.assert_close([0, 1, 2], x_min)
        math.assert_close([0, -1, -2], neg_x_min)

    def test_at_max_neighbor(self):
        x = math.range(spatial(x=4))
        x_min, neg_x_min = math.at_max_neighbor((x, -x), x, 'x')
        math.assert_close([1, 2, 3], x_min)
        math.assert_close([-1, -2, -3], neg_x_min)

    def test_index_shift(self):
        center, right = math.index_shift(math.meshgrid(x=3), (0, vec(x=1)), None)
        math.assert_close([0, 1], center['x'])
        math.assert_close([1, 2], right['x'])
        center, right = math.index_shift(math.meshgrid(x=3), (0, vec(x=1)), 'periodic')
        math.assert_close([0, 1, 2], center['x'])
        math.assert_close([1, 2, 0], right['x'])
        center, right = math.index_shift(math.meshgrid(x=3), (0, vec(x=2)), None)
        math.assert_close([0], center['x'])
        math.assert_close([2], right['x'])
        center, right, left = math.index_shift(math.meshgrid(x=3), (0, vec(x=1), vec(x=-1)), None)
        math.assert_close([1], center['x'])
        math.assert_close([2], right['x'])
        math.assert_close([0], left['x'])
        center, right, left = math.index_shift(math.meshgrid(x=3), (0, vec(x=1), vec(x=-1)), 'periodic')
        math.assert_close([0, 1, 2], center['x'])
        math.assert_close([1, 2, 0], right['x'])
        math.assert_close([2, 0, 1], left['x'])
        # --- 2D ---
        right, bottom = math.index_shift(math.meshgrid(x=3, y=3), (vec(x=1), vec(y=-1)), None)
        math.assert_close([[1, 1], [2, 2]], right['x'])
        math.assert_close([[1, 2], [1, 2]], right['y'])
        math.assert_close([[0, 0], [1, 1]], bottom['x'])
        math.assert_close([[0, 1], [0, 1]], bottom['y'])

    def test_index_shift_widths(self):
        from phiml.math._nd import index_shift_widths
        self.assertEqual([{'x': (0, 1)}, {'x': (1, 0)}], index_shift_widths([0, vec(x=1)]))

    def test_find_closest(self):
        for method in ['kd', 'dense']:
            vectors = wrap([(0, 0), (1, 0), (0, 1), (1, 1)], instance('vectors'), channel(vector='x,y'))
            lookup = vec(x=[.4, .6, .6], y=[.4, .4, .6])
            idx = math.find_closest(vectors, lookup, method=method)
            math.assert_close([(0,), (1,), (3,)], idx)
            math.assert_close([(0, 0), (1, 0), (1, 1)], vectors[idx])
            # --- multiple list dims ---
            vectors = wrap([[(0, 0), (1, 0)], [(0, 1), (1, 1)]], instance('vectors1,vectors2'), channel(vector='x,y'))
            lookup = vec(x=[.4, .6, .6], y=[.4, .4, .6])
            idx = math.find_closest(vectors, lookup, method=method)
            math.assert_close([(0, 0), (0, 1), (1, 1)], idx)
            math.assert_close([(0, 0), (1, 0), (1, 1)], vectors[idx])

    def test_find_closest_jit(self):
        find_closest = math.jit_compile(math.find_closest)
        for method in ['kd', 'dense']:
            vectors = wrap([(0, 0), (1, 0), (0, 1), (1, 1)], instance('vectors'), channel(vector='x,y'))
            lookup = vec(x=[.4, .6, .6], y=[.4, .4, .6])
            idx = find_closest(vectors, lookup, method=method)
            math.assert_close([(0,), (1,), (3,)], idx)
            math.assert_close([(0, 0), (1, 0), (1, 1)], vectors[idx])
            # --- multiple list dims ---
            vectors = wrap([[(0, 0), (1, 0)], [(0, 1), (1, 1)]], instance('vectors1,vectors2'), channel(vector='x,y'))
            lookup = vec(x=[.4, .6, .6], y=[.4, .4, .6])
            idx = find_closest(vectors, lookup, method=method)
            math.assert_close([(0, 0), (0, 1), (1, 1)], idx)
            math.assert_close([(0, 0), (1, 0), (1, 1)], vectors[idx])

    def test_clip_length(self):
        vecs = vec(x=[0, 1, 2, 3, 4, 5], y=0, z=0)
        clipped = math.clip_length(vecs, 2, 3)
        math.assert_close(0, clipped['y,z'])
        math.assert_close([0, 2, 2, 3, 3, 3], clipped['x'])
