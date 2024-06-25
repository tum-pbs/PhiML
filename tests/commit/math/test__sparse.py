from unittest import TestCase

from phiml import math
from phiml.backend._backend import init_installed_backends
from phiml.math import batch, get_sparsity, expand, wrap, stack, zeros, channel, spatial, ones, instance, tensor, \
    pairwise_distances, dense, assert_close, non_dual, dual, concat
from phiml.math._sparse import SparseCoordinateTensor, CompressedSparseMatrix
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

BACKENDS = init_installed_backends()


class TestSparse(TestCase):

    def test_sparse_tensor(self):
        indices = math.vec(x=[0, 1], y=[0, 1])
        values = math.ones(instance(indices))
        t = math.sparse_tensor(indices, values, spatial(x=2, y=2), format='coo')
        math.assert_close(wrap([[1, 0], [0, 1]], spatial('x,y')), t)

    def test_sparsity(self):
        self.assertEqual(1, get_sparsity(wrap(1)))
        self.assertEqual(0.25, get_sparsity(expand(1., batch(b=4))))
        self.assertEqual(0.25, get_sparsity(stack([zeros(batch(b=4))] * 3, channel('vector'))))
        self.assertEqual(0.3, get_sparsity(SparseCoordinateTensor(ones(instance(nnz=3), channel(vector='x'), dtype=int), ones(instance(nnz=3)), spatial(x=10), True, False, indices_constant=True)))
        self.assertEqual(0.03, get_sparsity(CompressedSparseMatrix(indices=ones(instance(nnz=3), dtype=int),
                                                                   pointers=ones(instance(y_pointers=11), dtype=int),
                                                                   values=ones(instance(nnz=3)),
                                                                   uncompressed_dims=spatial(x=10),
                                                                   compressed_dims=spatial(y=10),
                                                                   indices_constant=False)))

    def test_csr(self):
        for backend in BACKENDS:
            with backend:
                indices = tensor([0, 1, 0], instance('nnz'))
                pointers = tensor([0, 2, 3, 3], instance('pointers'))
                values = tensor([2, 3, 4], instance('nnz'))
                matrix = CompressedSparseMatrix(indices, pointers, values, channel(right=3), channel(down=3), 0)
                math.print(dense(matrix))
                assert_close((2, 3, 0), dense(matrix).down[0])
                assert_close((4, 0, 0), dense(matrix).down[1])
                assert_close((0, 0, 0), dense(matrix).down[2])
                # Multiplication
                assert_close((5, 4, 0), matrix.right * (1, 1, 1))
                # Simple arithmetic
                assert_close(matrix, (matrix + matrix * 2) / 3)

    def test_csr_slice_concat(self):
        pos = tensor([(0, 0), (0, 1), (0, 2)], instance('particles'), channel(vector='x,y'))
        dx = pairwise_distances(pos, max_distance=1.5, format='csr')
        self.assertEqual(0, dx.sum)
        dist = math.vec_length(dx, eps=1e-6)
        self.assertEqual(set(dual(particles=3) & instance(particles=3)), set(dist.shape))
        self.assertGreater(dist.sum, 0)
        # Slice channel
        dx_y = dx['y']
        self.assertEqual(set(dual(particles=3) & instance(particles=3)), set(dx_y.shape))
        # Slice / concat compressed
        concat_particles = math.concat([dx.particles[:1], dx.particles[1:]], 'particles')
        math.assert_close(dx, concat_particles)
        # Slice / concat uncompressed
        concat_dual = math.concat([dx.particles.dual[:1], dx.particles.dual[1:]], '~particles')
        math.assert_close(dx, concat_dual)

    def test_coo(self):
        def f(x):
            return math.laplace(x)

        for backend in BACKENDS:
            with backend:
                x = math.ones(spatial(x=5))
                coo, bias = math.matrix_from_function(f, x)
                csr = coo.compress(non_dual)
                math.assert_close(f(x), coo @ x, csr @ x)

    def test_ilu(self):
        def f(x):
            """
            True LU for the 3x3 matrix is

            L =  1  0  0    U = -2  1   0
               -.5  1  0         0 -1.5 1
                 0 -.7 1         0  0  -1.3
            """
            return math.laplace(x, padding=math.extrapolation.ZERO)
        matrix, bias = math.matrix_from_function(f, math.ones(spatial(x=5)))
        # --- Sparse ILU ---
        L, U = math.factor_ilu(matrix, 10)
        L, U = math.dense(L), math.dense(U)
        math.assert_close(L @ U, matrix)
        # --- Dense ILU ---
        matrix = math.dense(matrix)
        L, U = math.factor_ilu(matrix, 10)
        math.assert_close(L @ U, matrix)

    def test_tensor_like(self):
        matrix, _ = math.matrix_from_function(math.laplace, math.zeros(spatial(x=3)), padding=0)
        other = math.tensor_like(matrix, 1)
        math.assert_close(1, math.stored_values(other))
        math.assert_close(math.stored_indices(matrix), math.stored_indices(other))

    def test_boolean_mask(self):
        indices = math.vec(x=[0, 1], y=[0, 1])
        values = wrap([1, 2], instance(indices))
        t = math.sparse_tensor(indices, values, spatial(x=2, y=2), format='coo')  # (0, 1), (2, 0)
        mask = wrap([True, True], spatial('x'))
        math.assert_close(t, math.boolean_mask(t, 'x', mask), t[mask])
        mask = wrap([False, True], spatial('x'))
        math.assert_close(t.x[1:], math.boolean_mask(t, 'x', mask))

    def test_dense_to_sparse(self):
        for target_format in ['dense', 'coo', 'csr', 'csc']:
            value = tensor([[0, 1], [3, 0]], channel('c'), dual('d'))
            sp = math.to_format(value, target_format)
            math.assert_close(value, sp)
            self.assertEqual(target_format, math.get_format(sp))

    def test_reduce(self):
        for format in ['coo', 'csr', 'csc']:
            matrix = wrap([[0, 1], [-1, 2]], channel('c'), dual('d'))
            matrix = math.to_format(matrix, format)
            # --- partial reduction ---
            math.assert_close([-1, 3], math.sum(matrix, channel))
            math.assert_close([1, 1], math.sum(matrix, dual))
            math.assert_close([-1, 1], math.min(matrix, channel))
            math.assert_close([1, -1], math.min(matrix, dual))  # the 0 is not part of the matrix anymore
            math.assert_close([-1, 2], math.max(matrix, channel))
            math.assert_close([1, 2], math.max(matrix, dual))

    def test_sparse_sparse_mul(self):
        expected = wrap([[9, 1], [-3, 3]], channel('in'), dual('out'))
        for format in ['coo', 'csr', 'csc', 'dense']:
            a = math.to_format(tensor([[1, 2], [3, 0]], channel('in'), dual('red')), format)
            b = math.to_format(tensor([[-1, 1], [5, 0]], channel('red'), dual('out')), format)
            math.assert_close(expected, a @ b, msg=format)

    def test_stack_values(self):
        for format in ['coo', 'csr', 'csc']:
            matrix = wrap([[0, 1], [-1, 2]], channel('c'), dual('d'))
            matrix = math.to_format(matrix, format)
            stacked = stack([matrix, matrix*2], channel(vector='x,y'))
            self.assertTrue(math.is_sparse(stacked))
            self.assertEqual(type(stacked), type(matrix))

    def test_concat_values(self):
        for format in ['coo', 'csr', 'csc']:
            matrix = wrap([[0, 1], [-1, 2]], channel('c'), dual('d'))
            matrix = math.to_format(matrix, format)
            matrix = expand(matrix, channel(vector=1))
            stacked = concat([matrix, matrix*2], 'vector')
            self.assertTrue(math.is_sparse(stacked))
            self.assertEqual(type(stacked), type(matrix))

    def test_close(self):
        for format in ['coo', 'csr', 'csc']:
            m = wrap([[0, 1], [-1, 2]], channel('c'), dual('d'))
            m1 = math.to_format(m, format)
            m2 = m1 * 2 * .5
            math.assert_close(m1, m2)

    def test_matrix_rank(self):
        for b in BACKENDS:
            with b:
                for format in ['dense', 'coo', 'csr', 'csc']:
                    # --- rank 1 ---
                    matrix = tensor([[1, 2], [-1, -2]], dual('c'), channel('r'))
                    matrix = math.to_format(matrix, format)
                    matrix = stack([matrix, matrix * 2], batch('b'))
                    ranks = math.matrix_rank(matrix)
                    math.assert_close(1, ranks)
                    # --- rank 2 ---
                    matrix = tensor([[1, 2], [1, 0]], dual('c'), channel('r'))
                    matrix = math.to_format(matrix, format)
                    matrix = stack([matrix, matrix * 2], batch('b'))
                    ranks = math.matrix_rank(matrix)
                    math.assert_close(2, ranks)

    def test_wrap_sparse_scipy(self):
        for backend in BACKENDS:
            with backend:
                # --- int ---
                for scipy_type in [coo_matrix, csr_matrix, csc_matrix]:
                    M = scipy_type([[1, 0, 3], [0, 2, 0]])
                    t = tensor(M, channel('c') & dual)
                    math.assert_close([1, 0, 3], math.dense(t).c[0])
                    self.assertEqual(int, t.dtype.kind)
                    self.assertEqual(t.default_backend, backend)
                # --- bool ---
                for scipy_type in [coo_matrix, csr_matrix, csc_matrix]:
                    M = scipy_type([[True, False, True], [False, True, False]])
                    t = tensor(M, channel('c') & dual)
                    math.assert_close([True, False, True], math.dense(t).c[0])
                    self.assertEqual(bool, t.dtype.kind)
                    self.assertEqual(t.default_backend, backend)
