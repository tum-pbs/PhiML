import numbers
import os
import sys
from typing import Union, Optional, Tuple, Callable, Sequence

import numpy as np
import numpy.random
import scipy.signal
import scipy.sparse
from scipy.sparse import issparse, csr_matrix, coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve_triangular

from . import Backend, ComputeDevice
from ._backend import combined_dim, SolveResult, TensorType
from ._dtype import from_numpy_dtype, to_numpy_dtype, DType, FLOAT64, BOOL, COMPLEX128, INT32


class NumPyBackend(Backend):
    """Core Python Backend using NumPy & SciPy"""

    def __init__(self):
        if sys.platform != "win32" and sys.platform != "darwin":
            mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        else:
            mem_bytes = -1
        processors = os.cpu_count()
        cpu = ComputeDevice(self, "CPU", 'CPU', mem_bytes, processors, "", 'CPU')
        Backend.__init__(self, "numpy", [cpu], cpu)

    def prefers_channels_last(self) -> bool:
        return True

    seed = np.random.seed
    clip = staticmethod(np.clip)
    argmax = staticmethod(np.argmax)
    argmin = staticmethod(np.argmin)
    minimum = np.minimum
    maximum = np.maximum
    ones_like = staticmethod(np.ones_like)
    zeros_like = staticmethod(np.zeros_like)
    reshape = staticmethod(np.reshape)
    concat = staticmethod(np.concatenate)
    stack = staticmethod(np.stack)
    tile = staticmethod(np.tile)
    transpose = staticmethod(np.transpose)
    sqrt = np.sqrt
    exp = np.exp
    erf = scipy.special.erf
    sin = np.sin
    arcsin = np.arcsin
    cos = np.cos
    arccos = np.arccos
    tan = np.tan
    arctan = np.arctan
    arctan2 = staticmethod(np.arctan2)
    sinh = np.sinh
    arcsinh = np.arcsinh
    cosh = np.cosh
    arccosh = np.arccosh
    tanh = np.tanh
    arctanh = np.arctanh
    log = np.log
    log2 = np.log2
    log10 = np.log10
    isfinite = np.isfinite
    isnan = np.isnan
    isinf = np.isinf
    abs = np.abs
    sign = np.sign
    round = staticmethod(np.round)
    ceil = np.ceil
    floor = np.floor
    log_gamma = scipy.special.loggamma
    gamma_inc_l = scipy.special.gammainc
    gamma_inc_u = scipy.special.gammaincc
    shape = staticmethod(np.shape)
    staticshape = staticmethod(np.shape)
    imag = staticmethod(np.imag)
    real = staticmethod(np.real)
    conj = staticmethod(np.conjugate)
    einsum = staticmethod(np.einsum)
    cumsum = staticmethod(np.cumsum)

    def as_tensor(self, x, convert_external=True):
        if self.is_tensor(x, only_native=convert_external):
            array = x
        else:
            array = np.array(x)
        # --- Enforce Precision ---
        if not isinstance(array, numbers.Number):
            if self.dtype(array).kind == float:
                array = self.to_float(array)
            elif self.dtype(array).kind == complex:
                array = self.to_complex(array)
        return array

    def is_module(self, obj):
        return False

    def is_tensor(self, x, only_native=False):
        if isinstance(x, np.ndarray) and x.dtype != object and x.dtype != str:
            return True
        if issparse(x):
            return True
        if isinstance(x, (np.bool_, np.float32, np.float64, np.float16, np.int8, np.int16, np.int32, np.int64, np.complex128, np.complex64)):
            return True
        # --- Above considered native ---
        if only_native:
            return False
        # --- Non-native types
        if isinstance(x, (numbers.Number, bool)):
            return True
        if isinstance(x, (tuple, list)):
            return all([self.is_tensor(item, False) for item in x])
        return False

    def is_sparse(self, x) -> bool:
        return issparse(x)

    def get_sparse_format(self, x) -> str:
        format_names = {
            coo_matrix: 'coo',
            csr_matrix: 'csr',
            csc_matrix: 'csc',
        }
        return format_names.get(type(x), 'dense')

    def disassemble(self, x) -> Tuple[Callable, Sequence[TensorType]]:
        if issparse(x):
            if isinstance(x, coo_matrix):
                return lambda b, i, v: b.sparse_coo_tensor(i, v, x.shape), (np.stack([x.row, x.col], -1), x.data)
            if isinstance(x, csr_matrix):
                return lambda b, v, i, p: b.csr_matrix(i, p, v, x.shape), (x.data, x.indices, x.indptr)
            elif isinstance(x, csc_matrix):
                return lambda b, v, i, p: b.csc_matrix(p, i, v, x.shape), (x.data, x.indices, x.indptr)
            raise NotImplementedError
        else:
            return lambda b, t: t, (x,)

    def is_available(self, tensor):
        return True

    def numpy(self, tensor):
        if isinstance(tensor, np.ndarray) or issparse(tensor):
            return tensor
        else:
            return np.array(tensor)

    def numpy_call(self, f, output_shapes, output_dtypes, *args, **aux_args):
        output = f(*args, **aux_args)
        if isinstance(output_dtypes, DType):
            assert output.shape == output_shapes, f"numpy_call: output has shape {output.shape} but was promised to be {output_shapes}"
            assert self.dtype(output) == output_dtypes, f"{self.dtype(output)} != {output_dtypes}"
        else:
            assert len(output) == len(output_dtypes) == len(output_shapes)
            for i, (o, d, s) in enumerate(zip(output, output_dtypes, output_shapes)):
                assert len(o.shape) == len(s), f"numpy_call: out[{i}] has shape {o.shape} but was promised to be rank {len(s)}"
                assert all([s_ is None or o_ == s_ for o_, s_ in zip(o.shape, s)]), f"numpy_call: out[{i}] has shape {o.shape} but was promised to be {s}"
                assert self.dtype(o) == d, f"numpy_call: out[{i}] has dtype {o.dtype} but was promised to be {d}"
        return output

    def copy(self, tensor, only_mutable=False):
        return np.copy(tensor)

    def get_device(self, tensor) -> ComputeDevice:
        return self._default_device

    def allocate_on_device(self, tensor: TensorType, device: ComputeDevice) -> TensorType:
        assert device == self._default_device, f"NumPy Can only allocate on the CPU but got device {device}"
        return tensor

    def equal(self, x, y):
        if isinstance(x, np.ndarray) and x.dtype.char == 'U':  # string comparison
            x = x.astype(object)
        if isinstance(x, str):
            x = np.array(x, object)
        return np.equal(x, y)

    def divide_no_nan(self, x, y):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = x / y
        return np.where(y == 0, 0, result)

    def softplus(self, x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def random_uniform(self, shape, low, high, dtype: Optional[DType]):
        dtype = dtype or self.float_type
        if dtype.kind == float:
            return np.random.uniform(low, high, shape).astype(to_numpy_dtype(dtype))
        elif dtype.kind == complex:
            return (np.random.uniform(low.real, high.real, shape) + 1j * np.random.uniform(low.imag, high.imag, shape)).astype(to_numpy_dtype(dtype))
        elif dtype.kind == int:
            return numpy.random.randint(low, high, shape, dtype=to_numpy_dtype(dtype))
        else:
            raise ValueError(dtype)

    def random_normal(self, shape, dtype: DType):
        dtype = dtype or self.float_type
        return np.random.standard_normal(shape).astype(to_numpy_dtype(dtype))

    def random_permutations(self, permutations: int, n: int):
        return np.stack([np.random.permutation(n) for _ in range(permutations)])

    def random_subsets(self, element_count: int, subset_size: int, subset_count: int, allow_duplicates: bool, element_weights=None):
        if element_weights is not None:
            assert element_weights.ndim == 1
            element_weights /= element_weights.sum()
        return np.stack([np.random.choice(element_count, size=subset_size, replace=allow_duplicates, p=element_weights) for _ in range(subset_count)])

    def range(self, start, limit=None, delta=1, dtype: DType = INT32):
        if limit is None:
            start, limit = 0, start
        return np.arange(start, limit, delta, to_numpy_dtype(dtype))

    def pad(self, value, pad_width, mode='constant', constant_values=0):
        if mode not in ('constant', 'symmetric', 'periodic', 'reflect', 'boundary'):
            return NotImplemented
        if mode == 'constant':
            value, constant_values = self.auto_cast(value, constant_values)
            return np.pad(value, pad_width, 'constant', constant_values=constant_values)
        else:
            if mode in ('periodic', 'boundary'):
                mode = {'periodic': 'wrap', 'boundary': 'edge'}[mode]
            return np.pad(value, pad_width, mode)

    def sum(self, value, axis=None, keepdims=False):
        return np.sum(value, axis=axis, keepdims=keepdims)

    def prod(self, value, axis=None):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        if value.dtype == bool:
            return np.all(value, axis=axis)
        return np.prod(value, axis=axis)

    def where(self, condition, x=None, y=None):
        if x is None or y is None:
            return np.argwhere(condition)
        return np.where(condition, x, y)

    def nonzero(self, values, length=None, fill_value=-1):
        result = np.argwhere(values)
        if length is not None:
            result = self.pad_to(result, 0, length, fill_value)
        return result

    def zeros(self, shape, dtype: DType = None):
        return np.zeros(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def ones(self, shape, dtype: DType = None):
        return np.ones(shape, dtype=to_numpy_dtype(dtype or self.float_type))

    def meshgrid(self, *coordinates):
        return np.meshgrid(*coordinates, indexing='ij')

    def linspace(self, start, stop, number):
        return np.linspace(start, stop, number, dtype=to_numpy_dtype(self.float_type))

    def linspace_without_last(self, start, stop, number):
        return np.linspace(start, stop, number, dtype=to_numpy_dtype(self.float_type), endpoint=False)

    def mean(self, value, axis=None, keepdims=False):
        return np.mean(value, axis, keepdims=keepdims)

    def repeat(self, x, repeats, axis: int, new_length=None):
        return np.repeat(x, repeats, axis)

    def tensordot(self, a, a_axes: Union[tuple, list], b, b_axes: Union[tuple, list]):
        return np.tensordot(a, b, (a_axes, b_axes))

    def mul(self, a, b):
        if scipy.sparse.issparse(a):
            return a.multiply(b)
        elif scipy.sparse.issparse(b):
            return b.multiply(a)
        else:
            return Backend.mul(self, a, b)

    def mul_matrix_batched_vector(self, A, b):
        return np.stack([A.dot(b[i]) for i in range(b.shape[0])])

    def get_diagonal(self, matrices, offset=0):
        return np.transpose(np.diagonal(matrices, offset=offset, axis1=1, axis2=2), [0, 2, 1])

    def max(self, x, axis=None, keepdims=False):
        return np.max(x, axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        return np.min(x, axis, keepdims=keepdims)

    def conv(self, value, kernel, strides: Sequence[int], out_sizes: Sequence[int], transpose: bool):
        assert kernel.shape[0] in (1, value.shape[0])
        assert value.shape[1] == kernel.shape[2], f"value has {value.shape[1]} channels but kernel has {kernel.shape[2]}"
        assert value.ndim + 1 == kernel.ndim
        value, kernel = self.auto_cast(value, kernel, bool_to_int=True)
        has_strides = not all(st == 1 for st in strides)
        # --- Determine mode and pre-padding ---
        if not transpose:
            valid_size = [int(np.ceil((abs(vs - ks) + 1) / st)) for st, vs, ks in zip(strides, value.shape[2:], kernel.shape[3:])]
            same_size = [max(vs, ks) // st for st, vs, ks in zip(strides, value.shape[2:], kernel.shape[3:])]
            full_size = [(vs + ks - 1) // st for st, vs, ks in zip(strides, value.shape[2:], kernel.shape[3:])]
        else:
            kernel = np.flip(kernel, axis=tuple(range(3, kernel.ndim)))
            valid_size = [vs * st - ks + 1 for st, vs, ks in zip(strides, value.shape[2:], kernel.shape[3:])]  # ToDo
            same_size = [max(vs, ks) * st for st, vs, ks in zip(strides, value.shape[2:], kernel.shape[3:])]
            full_size = [(vs + ks - 1) * st for st, vs, ks in zip(strides, value.shape[2:], kernel.shape[3:])]
        needs_full = any(os > ss for os, ss in zip(out_sizes, same_size))
        needs_same = any(os > vs for os, vs in zip(out_sizes, valid_size))
        mode = 'full' if needs_full else ('same' if needs_same else 'valid')
        if any(os > fs for os, fs in zip(out_sizes, full_size)):
            raise NotImplementedError
        # --- Run conv for each input/output channel ---
        result = np.zeros((value.shape[0], kernel.shape[1], *out_sizes), dtype=to_numpy_dtype(self.float_type))
        if not transpose:
            for b in range(value.shape[0]):
                b_kernel = kernel[min(b, kernel.shape[0] - 1)]
                for o in range(kernel.shape[1]):
                    for i in range(value.shape[1]):
                        full = scipy.signal.correlate(value[b, i, ...], b_kernel[o, i, ...], mode=mode)
                        offset = [1 if ks >= 2 else 0 for os, ks, st in zip(value.shape[2:], kernel.shape[3:], strides)]  # 0 for ks=1, 1 for ks=2,3
                        result_o_i = full[tuple(slice(o, None, st) for o, st in zip(offset, strides))] if has_strides else full
                        # ToDo crop to fit out_sizes
                        result[b, o] += result_o_i
        else:
            upsampled = np.zeros([st * vs for st, vs in zip(strides, value.shape[2:])]) if has_strides else None  # Create a zero-initialized array with upsampled input
            for b in range(value.shape[0]):
                b_kernel = kernel[min(b, kernel.shape[0] - 1)]
                for o in range(kernel.shape[1]):
                    for i in range(value.shape[1]):
                        if has_strides:
                            upsampled[tuple([slice(0, None, st) for st in strides])] = value[b, i, ...]  # Place input values with stride
                        else:
                            upsampled = value[b, i, ...]
                        result_o_i = scipy.signal.correlate(upsampled, b_kernel[o, i, ...], mode=mode)
                        crop = [rs - os for rs, os in zip(result_o_i.shape, out_sizes)]
                        result[b, o] += result_o_i[tuple([slice((c+1)//2, -(c//2) or None) for c in crop])]
        if self.dtype(value).kind == int:
            result = result.astype(value.dtype)
        return result

    def expand_dims(self, a, axis=0, number=1):
        for _i in range(number):
            a = np.expand_dims(a, axis)
        return a

    def cast(self, x, dtype: DType):
        is_native = self.is_tensor(x, only_native=True)
        if is_native and from_numpy_dtype(x.dtype) == dtype:
            return x
        elif is_native:
            return x.astype(to_numpy_dtype(dtype))
        else:
            return np.array(x, to_numpy_dtype(dtype))

    def unravel_index(self, flat_index, shape):
        return np.stack(np.unravel_index(flat_index, shape), -1)

    def ravel_multi_index(self, multi_index, shape, mode: Union[str, int] = 'undefined'):
        mode = mode if isinstance(mode, int) else {'undefined': 'raise', 'periodic': 'wrap', 'clamp': 'clip'}[mode]
        idx_first = np.transpose(multi_index, (-1,) + tuple(range(self.ndims(multi_index)-1)))
        result = np.ravel_multi_index(idx_first, shape, mode='wrap' if isinstance(mode, int) else mode).astype(multi_index.dtype)
        if isinstance(mode, int):
            outside = self.any((multi_index < 0) | (multi_index >= shape), -1)
            result = self.where(outside, mode, result)
        return result

    def gather(self, values, indices, axis: int):
        slices = [indices if i == axis else slice(None) for i in range(self.ndims(values))]
        return values[tuple(slices)]

    def batched_gather_nd(self, values, indices):
        assert indices.shape[-1] == self.ndims(values) - 2
        batch_size = combined_dim(values.shape[0], indices.shape[0])
        indices_list = [indices[..., i] for i in range(indices.shape[-1])]
        batch_range = self.expand_dims(np.arange(batch_size), -1, number=self.ndims(indices) - 2)
        slices = (batch_range, *indices_list)
        return values[slices]

    def batched_gather_1d(self, values, indices):
        batch_size = combined_dim(values.shape[0], indices.shape[0])
        return values[np.arange(batch_size)[:, None], indices]

    def std(self, x, axis=None, keepdims=False):
        return np.std(x, axis, keepdims=keepdims)

    def boolean_mask(self, x, mask, axis=0, new_length=None, fill_value=0):
        slices = [mask if i == axis else slice(None) for i in range(len(x.shape))]
        result = x[tuple(slices)]
        if new_length is not None:
            result = self.pad_to(result, axis, new_length, fill_value)
        return result

    def any(self, boolean_tensor, axis=None, keepdims=False):
        return np.any(boolean_tensor, axis=axis, keepdims=keepdims)

    def all(self, boolean_tensor, axis=None, keepdims=False):
        return np.all(boolean_tensor, axis=axis, keepdims=keepdims)

    def scatter(self, base_grid, indices, values, mode: str):
        assert mode in ('add', 'update', 'max', 'min')
        assert isinstance(base_grid, np.ndarray)
        assert isinstance(indices, (np.ndarray, tuple))
        assert isinstance(values, np.ndarray)
        assert indices.ndim == 3
        assert values.ndim == 3
        assert base_grid.ndim >= 3
        batch_size = combined_dim(combined_dim(base_grid.shape[0], indices.shape[0]), values.shape[0])
        if base_grid.shape[0] == batch_size:
            result = np.copy(base_grid)
        else:
            result = np.tile(base_grid, (batch_size, *[1] * (base_grid.ndim - 1)))
        if not isinstance(indices, (tuple, list)):
            indices = self.unstack(indices, axis=-1)
        if mode == 'add':
            for b in range(batch_size):
                np.add.at(result, (b, *[i[min(b, i.shape[0]-1)] for i in indices]), values[min(b, values.shape[0]-1)])
        elif mode == 'update':
            for b in range(batch_size):
                result[(b, *[i[min(b, i.shape[0]-1)] for i in indices])] = values[min(b, values.shape[0]-1)]
        elif mode == 'max':
            for b in range(batch_size):
                np.maximum.at(result, (b, *[i[min(b, i.shape[0]-1)] for i in indices]), values[min(b, values.shape[0]-1)])
        elif mode == 'min':
            for b in range(batch_size):
                np.minimum.at(result, (b, *[i[min(b, i.shape[0]-1)] for i in indices]), values[min(b, values.shape[0]-1)])
        return result

    def histogram1d(self, values, weights, bin_edges):
        batch_size, value_count = self.staticshape(values)
        result = []
        for b in range(batch_size):
            hist, _ = np.histogram(values[b], bins=bin_edges[b], weights=weights[b])
            result.append(hist)
        return np.stack(result)

    def bincount(self, x, weights: Optional[TensorType], bins: int, x_sorted=False):
        result = np.bincount(x, weights=weights, minlength=bins)
        assert result.shape[-1] == bins
        return result

    def unique(self, x: TensorType, return_inverse: bool, return_counts: bool, axis: int) -> Tuple[TensorType, ...]:
        return np.unique(x, return_inverse=return_inverse, return_counts=return_counts, axis=axis)

    def quantile(self, x, quantiles):
        return np.quantile(x, quantiles, axis=-1)

    def argsort(self, x, axis=-1):
        return np.argsort(x, axis)

    def sort(self, x, axis=-1):
        return np.sort(x, axis)

    def searchsorted(self, sorted_sequence, search_values, side: str, dtype=INT32):
        if self.ndims(sorted_sequence) == 1:
            return np.searchsorted(sorted_sequence, search_values, side=side).astype(to_numpy_dtype(dtype))
        else:
            return np.stack([self.searchsorted(seq, val, side, dtype) for seq, val in zip(sorted_sequence, search_values)])

    def fft(self, x, axes: Union[tuple, list]):
        x = self.to_complex(x)
        if not axes:
            return x
        if len(axes) == 1:
            return np.fft.fft(x, axis=axes[0]).astype(x.dtype)
        elif len(axes) == 2:
            return np.fft.fft2(x, axes=axes).astype(x.dtype)
        else:
            return np.fft.fftn(x, axes=axes).astype(x.dtype)

    def ifft(self, k, axes: Union[tuple, list]):
        if not axes:
            return k
        if len(axes) == 1:
            return np.fft.ifft(k, axis=axes[0]).astype(k.dtype)
        elif len(axes) == 2:
            return np.fft.ifft2(k, axes=axes).astype(k.dtype)
        else:
            return np.fft.ifftn(k, axes=axes).astype(k.dtype)

    def dtype(self, array) -> DType:
        if isinstance(array, bool):
            return BOOL
        if isinstance(array, int):
            return INT32
        if isinstance(array, float):
            return FLOAT64
        if isinstance(array, complex):
            return COMPLEX128
        if not hasattr(array, 'dtype'):
            array = np.array(array)
        return from_numpy_dtype(array.dtype)

    def indexed_segment_sum(self, x, indices, axis: int):
        return np.stack([np.add.reduceat(x[b], indices[b], axis-1) for b in range(x.shape[0])])

    def sparse_coo_tensor(self, indices, values, shape):
        indices = self.unstack(indices, -1)
        if len(shape) == 2:
            return coo_matrix((values, indices), shape=shape)
        else:
            raise NotImplementedError(f"len(indices) = {len(indices)} not supported. Only (2) allowed.")

    def csr_matrix(self, column_indices, row_pointers, values, shape: tuple):
        return csr_matrix((values, column_indices, row_pointers), shape=shape)

    def mul_csr_dense(self, column_indices, row_pointers, values, shape: tuple, dense):
        batch_size, nnz, channel_count = values.shape
        result = []
        for b in range(batch_size):
            b_result = []
            for c in range(channel_count):
                mat = csr_matrix((values[b, :, c], column_indices[b], row_pointers[b]), shape=shape)
                b_result.append((mat * dense[b, :, c, :]))
            result.append(np.stack(b_result, 1))
        return np.stack(result)

    def csc_matrix(self, column_pointers, row_indices, values, shape: tuple):
        return csc_matrix((values, row_indices, column_pointers), shape=shape)

    def stop_gradient(self, value):
        return value

    # def jacobian(self, f, wrt: Union[tuple, list], get_output: bool):
    #     warnings.warn("NumPy does not support analytic gradients and will use differences instead. This may be slow!", RuntimeWarning)
    #     eps = {64: 1e-9, 32: 1e-4, 16: 1e-1}[self.precision]
    #
    #     def gradient(*args, **kwargs):
    #         output = f(*args, **kwargs)
    #         loss = output[0] if isinstance(output, (tuple, list)) else output
    #         grads = []
    #         for wrt_ in wrt:
    #             x = args[wrt_]
    #             assert isinstance(x, np.ndarray)
    #             if x.size > 64:
    #                 raise RuntimeError("NumPy does not support analytic gradients. Use PyTorch, TensorFlow or Jax.")
    #             grad = np.zeros_like(x).flatten()
    #             for i in range(x.size):
    #                 x_flat = x.flatten()  # makes a copy
    #                 x_flat[i] += eps
    #                 args_perturbed = list(args)
    #                 args_perturbed[wrt_] = np.reshape(x_flat, x.shape)
    #                 output_perturbed = f(*args_perturbed, **kwargs)
    #                 loss_perturbed = output_perturbed[0] if isinstance(output, (tuple, list)) else output_perturbed
    #                 grad[i] = (loss_perturbed - loss) / eps
    #             grads.append(np.reshape(grad, x.shape))
    #         if get_output:
    #             return output, grads
    #         else:
    #             return grads
    #     return gradient

    def linear_solve(self, method: str, lin, y, x0, rtol, atol, max_iter, pre, matrix_offset) -> SolveResult:
        if method in ['direct', 'CG-native', 'GMres', 'biCG', 'biCG-stab', 'CGS', 'lGMres', 'minres', 'QMR', 'GCrotMK'] and max_iter.shape[0] == 1:
            from ._linalg import scipy_sparse_solve
            return scipy_sparse_solve(self, method, lin, y, x0, rtol, atol, max_iter, pre, matrix_offset)
        return Backend.linear_solve(self, method, lin, y, x0, rtol, atol, max_iter, pre, matrix_offset)

    def matrix_solve_least_squares(self, matrix: TensorType, rhs: TensorType) -> TensorType:
        solution, residuals, rank, singular_values = [], [], [], []
        for b in range(self.shape(rhs)[0]):
            solution_b, residual_b, rnk_b, s_b = np.linalg.lstsq(matrix[b], rhs[b], rcond=None)
            solution.append(solution_b)
            residuals.append(residual_b)
            rank.append(rnk_b)
            singular_values.append(s_b)
        return np.stack(solution), np.stack(residuals), np.stack(rank), np.stack(singular_values)

    def solve_triangular_dense(self, matrix, rhs, lower: bool, unit_diagonal: bool):
        if matrix.ndim == 2:
            return scipy.linalg.solve_triangular(matrix, rhs, lower=lower, unit_diagonal=unit_diagonal)
        else:
            batch_size = matrix.shape[0]
            return np.stack([self.solve_triangular(matrix[b], rhs[b], lower, unit_diagonal) for b in range(batch_size)])

    def solve_triangular_sparse(self, matrix, rhs, lower: bool, unit_diagonal: bool):  # needs to be overridden to indicate this is natively implemented
        return spsolve_triangular(matrix, rhs.T, lower=lower, unit_diagonal=unit_diagonal).T

    def matrix_rank_dense(self, matrix, hermitian=False):
        return np.linalg.matrix_rank(matrix, hermitian=hermitian)

    def eigvals(self, matrix: TensorType) -> TensorType:
        return numpy.linalg.eigvals(matrix)

    def eig(self, matrix: TensorType) -> TensorType:
        return numpy.linalg.eig(matrix)

    def svd(self, matrix: TensorType, full_matrices=True) -> Tuple[TensorType, TensorType, TensorType]:
        result = np.linalg.svd(matrix, full_matrices=full_matrices)
        return result[0], result[1], result[2]


NUMPY = NumPyBackend()
