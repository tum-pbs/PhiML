import operator
import warnings
from collections import namedtuple
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Tuple, Union, Optional, List

from . import _ops as math
from ._magic_ops import stack, expand, rename_dims, value_attributes, pack_dims, concat
from ._nd import vec
from ._ops import backend_for, concat_tensor, scatter
from ._shape import Shape, merge_shapes, instance, EMPTY_SHAPE, dual, channel, batch, DEBUG_CHECKS, non_channel
from ._sparse import SparseCoordinateTensor, is_sparse, sparse_dims, sparse_tensor, stored_indices
from ._tensors import Tensor, wrap, TensorStack, BlockTensor, NO_OFFSET, IndexOffset, variable_dim_names, variable_shape, TensorProperties
from ._tree import disassemble_tree, assemble_tree
from .extrapolation import Extrapolation, ConstantExtrapolation
from ..backend import NUMPY, Backend
from ..backend._dtype import DType

TracerSource = namedtuple('TracerSource', ['shape', 'dtype', 'name', 'index', 'example_value'])


@dataclass(frozen=True, eq=False, unsafe_hash=False, repr=False)
class LinTracer(Tensor):
    """ Uniform `Tensor` where each output value depends on a fixed number of input values.
    These dependencies are listed in `_fac` along `_deps:b`.
    The shape of LinTracer is equal to the `_bias` shape.
    """
    _source: TracerSource
    _indices: Tensor
    """ Shape compatible with self.shape. Always includes all dims of `_fac`.
    Special dims:
        * channel dim 'idx': contains only relevant source dims, other src dim dependence is constant.
        * batch dim `_deps': contributions from multiple input indices to be summed
    """
    _fac: Tensor
    """ multiplication factors: sum(mul * src[indices], '_deps'). Can have fewer dims than indices. Shape compatible with self.shape. May contains _deps."""
    _fac_nz: Tensor[bool]
    """ NumPy-backed mask whose shape is fully contained in `_fac`. `False` values indicate where `_fac` is guaranteed to be zero. """
    _bias: Tensor
    """ Shape equal to `self.shape`, dtype equal to `self.dtype`. Can contain additional expanded dims along which values (not dependencies) are constant """

    @classmethod
    def create_identity(cls, src: TracerSource):
        indices = math.zeros(batch(_deps=1), channel(idx=''), dtype=int)
        fac = wrap(1)
        fac_nz = wrap(True)
        bias = math.zeros(src.shape, dtype=src.dtype)
        return cls(src, indices, fac, fac_nz, bias)

    def __post_init__(self):
        assert self._fac.shape & self._indices.shape  # Shapes must be broadcastable
        assert self._fac.shape in self._indices  # _indices must include all dims of _fac
        assert self._fac_nz.shape in self._fac.shape
        assert self._fac_nz.backend == NUMPY
        assert self._fac_nz.dtype.kind == bool
        assert self._indices.dtype.kind == int
        if DEBUG_CHECKS:
            # check all indices within bounds
            assert self._indices.min >= 0, f"Encountered negative index"
            for comp in self._indices.shape.get_labels('idx'):
                c_indices = self._indices.idx[comp]
                assert c_indices.max < self._source.shape.get_size(comp), f"Invalid index encountered for component {comp}: {c_indices.max} but size is {self._source.shape.get_size(comp)}"

    def _source_indices(self, included_src_dims: Shape, order: Tuple[str] = None, as_dual=False):
        """
        Args:
            included_src_dims: Dim names in `self._source.shape` that should be part of the result's shape even if the dependency is constant along them.
        """
        if order is not None:
            assert all([name in order for name in self._indices.shape['idx'].labels[0]]), f"All dependent dims ({self._indices.shape['idx'].labels[0]}) must be listed in included_src_dims but got {included_src_dims}"
        extend = []  # index components to add because of included_dims
        constant_dims = []
        with NUMPY:
            for dim in included_src_dims - self._var_src_names:  # dims that are not yet in self._indices, constant dependence (diagonal)
                assert dim in self.shape, f"Cannot add source dim {dim} after it has been sliced off (not in self.shape)"
                extend.append(vec('idx', **{dim.name: math.arange(self._bias.shape[dim.name])}))
        as_primal = concat([self._indices, *extend], 'idx', expand_values=True)
        as_primal = expand(as_primal, *constant_dims)
        if order is not None and as_primal.shape.get_labels('idx') != order:
            as_primal = as_primal.idx[order]
        if as_dual:
            dual_shape = as_primal.shape.with_dim_size('idx', ['~' + label if not label.startswith('~') else label for label in as_primal.shape['idx'].slice_names])
            return as_primal._with_shape_replaced(dual_shape)
        else:
            return as_primal

    @property
    def shape(self) -> Shape:
        return self._bias.shape

    @property
    def dtype(self) -> DType:
        return self._bias.dtype

    @property
    def backend(self) -> Backend:
        return backend_for(self._bias, self._fac)

    @property
    def _is_tracer(self) -> bool:
        return True

    @property
    def _var_dims(self) -> Tuple[str, ...]:
        return tuple(set(self._bias._var_dims) | (set(self._indices.shape.names) - {'idx', '_deps'}))

    @property
    def _var_lin_dims(self):
        return set(self._indices.shape.names) - {'idx', '_deps'}

    @property
    def _var_src_names(self):
        return self._indices.shape['idx'].labels[0]

    def native(self, order: Union[str, tuple, list, Shape] = None, force_expand=True):
        raise NotImplementedError

    def _with_shape_replaced(self, new_shape: Shape):
        changed = merge_shapes(*[o for o, n in zip(self.shape, new_shape) if o != n])
        new_dims = self._source.shape.only(changed) - (self._indices.shape - 'idx' - '_deps')
        indices = self._source_indices(included_src_dims=new_dims)  # make sure that changed dims are stored in indices
        indices = indices._with_shape_replaced(indices.shape.replace(self.shape, new_shape))
        fac = self._fac._with_shape_replaced(self._fac.shape.replace(self.shape, new_shape))
        fac_nz = self._fac_nz._with_shape_replaced(self._fac_nz.shape.replace(self.shape, new_shape))
        bias = self._bias._with_shape_replaced(new_shape)
        return LinTracer(self._source, indices, fac, fac_nz, bias)

    def _getitem(self, selection: dict) -> 'Tensor':
        new_dims = self._source.shape.only(tuple(selection)) - self._indices.shape
        indices = self._source_indices(included_src_dims=new_dims)[selection]
        fac = self._fac[selection]
        fac_nz = self._fac_nz[selection]
        bias = self._bias[selection]
        return LinTracer(self._source, indices, fac, fac_nz, bias)

    def _unstack(self, dimension: str):
        dim = self.shape[dimension]
        return tuple([self[{dimension: i}] for i in range(dim.size)])

    def _op2(self, other, op: Callable, switch_args: bool) -> Tensor:
        if is_sparse(other):
            return NotImplemented
        if isinstance(other, (TensorStack, BlockTensor)):
            return NotImplemented
        assert op in {operator.add, operator.sub, operator.mul, operator.truediv}, f"Unsupported operation encountered while tracing linear function: {op}"
        if isinstance(other, Tensor) and other._is_tracer:
            assert op in {operator.add, operator.sub}, f"Non-linear tracer-tracer operation encountered while tracing linear function: {op}"
            if op == operator.add:
                t1, t2 = self, other
            else:  # sub
                t1, t2 = (-self, other) if switch_args else (self, -other)
            # --- Add uniform tracers ---
            if isinstance(other, LinTracer):
                src_dims = dependent_src_dims(t1) & dependent_src_dims(t2)
                idx1 = t1._source_indices(included_src_dims=src_dims, order=src_dims.names)
                idx2 = t2._source_indices(included_src_dims=src_dims, order=src_dims.names)
                indices = concat([idx1, idx2], '_deps', expand_values=True)
                fac1, fac2 = [t._fac if '_deps' in t._fac.shape else expand(t._fac, batch(_deps=1)) for t in (t1, t2)]
                fac = concat([fac1, fac2], '_deps', expand_values=True)
                fac_nz1, fac_nz2 = [t._fac_nz if '_deps' in t._fac_nz.shape else expand(t._fac_nz, batch(_deps=1)) for t in (t1, t2)]
                fac_nz = concat([fac_nz1, fac_nz2], '_deps', expand_values=True)
                bias = t1._bias + t2._bias
                return LinTracer(self._source, indices, fac, fac_nz, bias)
            return BlockTensor(t1.shape & t2.shape, [(t1, NO_OFFSET), (t2, NO_OFFSET)], operator.add)
        else:  # op with constant
            other = self._tensor(other)
            bias = op(self._bias, other)
            if op in {operator.mul, operator.truediv}:
                fac = op(self._fac, other)
                fac_nz = self._fac_nz
                if op == operator.mul and other.available:
                    fac_nz = fac_nz & math.convert(other != 0, NUMPY)
                indices = self._source_indices(included_src_dims=self._source.shape.only(variable_shape(other)))
                indices = expand(indices, fac.shape)
                return LinTracer(self._source, indices, fac, fac_nz, bias)
            elif op in {operator.add, operator.sub}:
                return LinTracer(self._source, self._indices, self._fac, self._fac_nz, bias)
            else:
                raise ValueError(f"Unsupported operation encountered while tracing linear function: {op}")

    def _op1(self, native_function, op_name: str) -> Tensor:
        # __neg__ and __cast__ implemented below
        if native_function.__name__ == 'isfinite':
            return expand(math.is_finite(self._fac), self.shape)
        elif op_name == 'neg':
            return self.__neg__()
        elif op_name in {'cast', 'to_float', 'to_int32', 'to_int64', 'to_complex'}:
            raise AssertionError("cast called via _op1. Should be __cast__ instead")
        else:
            raise NotImplementedError('Only linear operations are supported')

    def __neg__(self):
        return LinTracer(self._source, self._indices, -self._fac, self._fac_nz, -self._bias)

    def __cast__(self, dtype: DType) -> 'Tensor':
        if self.dtype == dtype:
            return self
        if self._source.dtype & dtype == self.dtype:  # cannot down-cast
            warnings.warn(f"Cannot cast linear tracer of type {self.dtype} to {dtype} because its input has type {self._source.dtype}", RuntimeWarning)
            return self
        fac = math.cast(self._fac, dtype)
        bias = math.cast(self._bias, dtype)
        return LinTracer(self._source, self._indices, fac, self._fac_nz, bias)

    def _natives(self) -> tuple:
        """ This function should only be used to determine the compatible backends, this tensor should be regarded as not available. """
        return self._fac._natives()

    def _spec_dict(self) -> dict:
        raise LinearTraceInProgress(self)

    def _sum(self, dims: Shape):
        new_dims = self._source.shape.only(dims) - self._indices.shape
        indices = self._source_indices(included_src_dims=new_dims)
        indices = pack_dims(indices, ['_deps', dims], '_deps:b')
        fac = pack_dims(self._fac, ['_deps', dims], '_deps:b')
        fac_nz = pack_dims(self._fac_nz, ['_deps', dims], '_deps:b')
        bias = math.sum_(self._bias, dims)
        return LinTracer(self._source, indices, fac, fac_nz, bias)

    def _gather(self, indices: Tensor):
        """
        Args:
            indices: has 1 channel and 1 non-channel/non-instance
        """
        dims = channel(indices).labels[0]
        bias = self._bias[indices]
        new_dims = self._source.shape.only(dims) - self._indices.shape
        idx = self._source_indices(included_src_dims=new_dims)[indices]
        fac = self._fac[indices]
        fac_nz = self._fac_nz[indices]
        return LinTracer(self._source, idx, fac, fac_nz, bias)

    def _dot(self, self_dims: Shape, matrix: Tensor, matrix_dims: Shape) -> Tensor:
        t1 = rename_dims(self, self_dims, channel('_reduce'))
        t2 = rename_dims(matrix, matrix_dims, channel('_reduce'))
        mul = t1 * t2
        return mul._sum('_reduce')

    def _scatter(self, base_grid: Tensor, indices: Tensor, mode: str, index_dim: Shape, indexed_dims: Shape, batches: Shape, channels: Shape, lists: Shape) -> Tensor:
        # This function may be called by dense(SparseTensor) and must return a dense tensor
        if base_grid._is_tracer:
            raise NotImplementedError
        if mode == 'update':  # max dependencies unchanged -> return dense LinTracer
            lin_indices = scatter(expand(0, indexed_dims), indices, self._indices)
            fac = scatter(expand(0, indexed_dims), indices, self._fac)
            fac_nz = scatter(expand(False, indexed_dims), indices, self._fac_nz)
            bias = scatter(base_grid, indices, self._bias)
            return LinTracer(self._source, lin_indices, fac, fac_nz, bias)
        elif mode == 'add':  # With duplicate indices, we can get more dependencies in the output
            tr_indices, fac, fac_nz = math.bins((self._indices, self._fac, self._fac_nz), indices, instance, bins=base_grid.shape.only(channel(indices).labels[0]), bin_dim=batch('_dupli'))
            tr_indices = pack_dims(tr_indices, '_dupli,_deps', '_deps:b')
            fac = pack_dims(fac, '_dupli,_deps', '_deps:b')
            fac_nz = pack_dims(fac_nz, '_dupli,_deps', '_deps:b')
            bias = scatter(base_grid, indices, self._bias, mode=mode)
            return LinTracer(self._source, tr_indices, fac, fac_nz, bias)
        else:
            raise NotImplementedError

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **_kwargs) -> 'Tensor':
        if any(not isinstance(v, LinTracer) for v in values):
            return TensorStack([wrap(v) for v in values], dim)
        if len(values) == 1:
            return values[0].__expand__(dim)
        if merge_shapes(*values, allow_varying_sizes=True).undefined:
            return TensorStack(values, dim)
        src_dims = merge_shapes(*[dependent_src_dims(t) for t in values])
        indices = [t._source_indices(included_src_dims=src_dims, order=src_dims.names) for t in values if isinstance(t, LinTracer)]
        indices = stack(indices, dim, expand_values=True, **_kwargs)
        fac = stack([t._fac for t in values], dim, expand_values=True)
        fac_nz = stack([t._fac_nz for t in values], dim, expand_values=True)
        bias = stack([t._bias for t in values], dim)
        return LinTracer(values[0]._source, indices, fac, fac_nz, bias)

    def __expand__(self, dims: Shape, **kwargs) -> 'Tensor':
        return LinTracer(self._source, self._indices, self._fac, self._fac_nz, expand(self._bias, dims))

    def __pack_dims__(self, dims: Shape, packed_dim: Shape, pos: Union[int, None], **kwargs) -> 'Tensor':
        assert '_deps' not in dims
        new_dims = self._source.shape.only(dims) - self._indices.shape
        indices = self._source_indices(included_src_dims=new_dims)
        indices = indices.__pack_dims__(dims, packed_dim, pos)
        fac = expand(self._fac, dims).__pack_dims__(dims, packed_dim, pos)
        fac_nz = expand(self._fac_nz, dims).__pack_dims__(dims, packed_dim, pos)
        bias = self._bias.__pack_dims__(dims, packed_dim, pos)
        return LinTracer(self._source, indices, fac, fac_nz, bias)

    def _pad(self, ext: Extrapolation, widths, already_padded, **kwargs):
        no_bias: Extrapolation = ext - ext  # ToDo for constant extrapolation, return a composite tensor, so we don't have to filter out zero-values later (which may be impossible when jit-compiling)
        indices = self._source_indices(included_src_dims=self._source.shape.only(tuple(widths)))
        indices = no_bias.pad(indices, widths, already_padded, **kwargs)
        fac = no_bias.pad(expand(self._fac, (self._indices.shape & (self._source.shape - self._indices.shape)).only(tuple(widths)) - self._fac.shape), widths, already_padded, **kwargs)
        nz_ext = not isinstance(ext, ConstantExtrapolation)
        fac_nz = ConstantExtrapolation(nz_ext).pad(self._fac_nz, widths, already_padded, **kwargs)
        bias = ext.pad(self._bias, widths, already_padded, **kwargs)
        return LinTracer(self._source, indices, fac, fac_nz, bias)

    @staticmethod
    def __concat__(values: tuple, dim: str, **kwargs) -> 'Tensor':
        any_tracer = next(iter([t for t in values if isinstance(t, LinTracer)]))
        full_size = sum([t_.shape.get_size(dim) for t_ in values])
        max_dep_count = max([v._indices.shape.get_size('_deps') for v in values if isinstance(v, LinTracer)])
        src_dims = merge_shapes(*[dependent_src_dims(t) for t in values])
        if dim in src_dims:
            src_dims = src_dims.with_dim_size(dim, None)
        indices_list = []
        fac_list = []
        fac_nz_list = []
        bias_list = []
        for t in values:
            if isinstance(t, LinTracer):
                if t._indices.shape.get_size('_deps') < max_dep_count:
                    raise NotImplementedError
                indices_list.append(t._source_indices(included_src_dims=src_dims))
                fac_list.append(expand(t._fac, t.shape[dim]))
                fac_nz_list.append(expand(t._fac_nz, t.shape[dim]))
                bias_list.append(t._bias)
            else:
                indices_list.append(math.zeros(t.shape[dim], dtype=int))
                fac_list.append(math.zeros(t.shape[dim], dtype=bool))
                fac_nz_list.append(math.zeros(t.shape[dim], dtype=bool))
                bias_list.append(t)
        indices = concat(indices_list, dim, expand_values=True)
        fac = concat(fac_list, dim, expand_values=True)
        fac_nz = concat(fac_nz_list, dim, expand_values=True)
        bias = concat(bias_list, dim)
        return LinTracer(any_tracer._source, indices, fac, fac_nz, bias)

    def _simplify(self):
        # --- Check if dim can be dropped because nothing is done to it at the end of the day ---
        unnecessary = set()
        for dim in self.shape:
            if dim.name in self._var_src_names and dim not in self._fac.shape and dim.size == self._source.shape.get_size(dim):
                indices = self._indices.idx[dim.name]
                trivial = math.arange(dim)
                if (indices == trivial).all:
                    unnecessary.add(dim.name)
        if not unnecessary:
            return self
        keep = [dim for dim in self._var_src_names if dim not in unnecessary]
        indices = self._indices.idx[keep]
        indices = indices[{u: 0 for u in unnecessary}]  # indices must be repeated along unnecessary dims
        return LinTracer(self._source, indices, self._fac, self._fac_nz, self._bias)

    def __repr__(self):
        try:
            example_value = self.example_value
        except Exception as exc:
            example_value = str(exc)
        if self._indices.shape.volume > 0:
            return f"{self._bias.shape} up to {self._indices.shape.get_size('_deps')} lin deps per entry from {self._source.shape}. Example value: {example_value}"
        else:
            return f"{self._bias.shape} lin identity from {self._source.shape}. Example value: {example_value}"

    def _debug_print_dependencies(self, idx=None):
        if idx is None:
            idx = next(iter(self.shape.meshgrid()))
        print(f"Linear dependencies at {idx}:", end=" ")
        indices = self._indices[idx]
        fac = self._fac[idx]
        for i in (indices.shape['_deps']).meshgrid():
            print(f"{fac[i]} at {indices[i]}", end=", ")
        print()

    def _apply(self, source_value: Tensor):
        if channel(self._indices).size == 0:  # nothing to gather
            return self._source.example_value + self._bias
        indices = self._source_indices(included_src_dims=self._source.shape)
        return math.sum_(math.gather(source_value, indices, pref_index_dim='idx') * self._fac, '_deps') + self._bias

    @property
    def example_value(self):
        return self._apply(self._source.example_value)


@dataclass(frozen=True, eq=False, unsafe_hash=False, repr=False)
class DebugLinTracer(LinTracer):
    """LinTracer subclass that wraps all inherited methods so that any returned LinTracer is replaced by a DebugLinTracer with equal values."""

    @staticmethod
    def _debug_check(fun, example_fun, *args, **kwargs):
        result = fun(*args, **kwargs)
        if type(result) is not LinTracer:
            return result
        example_args = [a.example_value if isinstance(a, LinTracer) else a for a in args]
        example_kwargs = {k: v.example_value if isinstance(v, LinTracer) else v for k, v in kwargs.items()}
        example_result = example_fun(*example_args, **example_kwargs)
        if not math.close(result.example_value, example_result, abs_tolerance=1e-5, rel_tolerance=1e-5, equal_nan=True):
            warnings.warn(
                f"Linear trace inconsistency detected in function '<tracer>.{fun.__name__}()'. Breakpoint triggered. Running function again for manual debugging.\nExample result: {example_result}\nTraced result: {result.example_value}")
            breakpoint()
            fun(*args, **kwargs)
        return DebugLinTracer(result._source, result._indices, result._fac, result._fac_nz, result._bias)

    def _with_shape_replaced(self, new_shape: Shape):
        order = self.example_value.shape.indices(self.shape.names)
        return self._debug_check(LinTracer._with_shape_replaced, lambda example, new_shape: example._with_shape_replaced(new_shape[order]), self, new_shape)

    def _getitem(self, selection: dict) -> 'Tensor':
        return self._debug_check(LinTracer._getitem, lambda example, selection: example._getitem(selection), self, selection=selection)

    def _unstack(self, dimension: str):
        return self._debug_check(LinTracer._unstack, lambda example, dim: example._unstack(dim), self, dimension=dimension)

    def _op2(self, other, op: Callable, switch_args: bool) -> Tensor:
        return self._debug_check(LinTracer._op2, lambda example, other, op, switch_args: example._op2(other, op, switch_args), self, other, op, switch_args=switch_args)

    def _op1(self, native_function, op_name: str) -> Tensor:
        return self._debug_check(LinTracer._op1, lambda example, native_function, op_name: example._op1(native_function, op_name), self, native_function, op_name=op_name)

    def __neg__(self):
        return self._debug_check(LinTracer.__neg__, lambda example: -example, self)

    def __cast__(self, dtype: DType) -> 'Tensor':
        return self._debug_check(LinTracer.__cast__, lambda example, dtype: math.cast(example, dtype), self, dtype=dtype)

    def _sum(self, dims: Shape):
        return self._debug_check(LinTracer._sum, lambda example, dims: math.sum_(example, dims), self, dims=dims)

    def _gather(self, indices: Tensor):
        return self._debug_check(LinTracer._gather, lambda example, indices: math.gather(example, indices), self, indices=indices)

    def _dot(self, self_dims: Shape, matrix: Tensor, matrix_dims: Shape) -> Tensor:
        return self._debug_check(LinTracer._dot, lambda example, self_dims, matrix, matrix_dims: example._dot(self_dims, matrix, matrix_dims), self, self_dims, matrix, matrix_dims=matrix_dims)

    def _scatter(self, base_grid: Tensor, indices: Tensor, mode: str, index_dim: Shape, indexed_dims: Shape, batches: Shape, channels: Shape, lists: Shape) -> Tensor:
        def example_scatter(example: Tensor, base_grid: Tensor, indices: Tensor, mode: str, *_args):
            return math.scatter(base_grid, indices, example, mode)
        return self._debug_check(LinTracer._scatter, example_scatter, self, base_grid, indices, mode, index_dim, indexed_dims, batches, channels, lists)

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **_kwargs) -> 'Tensor':
        def example_stack(values, dim, **kw):
            example_values = tuple(v.example_value if isinstance(v, LinTracer) else v for v in values)
            return stack(example_values, dim, **kw)
        return DebugLinTracer._debug_check(LinTracer.__stack__, example_stack, values, dim, **_kwargs)

    def __expand__(self, dims: Shape, **kwargs) -> 'Tensor':
        return self._debug_check(LinTracer.__expand__, lambda example, dims, **kw: expand(example, dims, **kw), self, dims=dims, **kwargs)

    def __pack_dims__(self, dims: Shape, packed_dim: Shape, pos: Union[int, None], **kwargs) -> 'Tensor':
        return self._debug_check(LinTracer.__pack_dims__, lambda example, dims, packed_dim, pos, **kw: example.__pack_dims__(dims, packed_dim, pos, **kw), self, dims, packed_dim, pos=pos, **kwargs)

    def _pad(self, ext: Extrapolation, widths, already_padded, **kwargs):
        return self._debug_check(LinTracer._pad, lambda example, ext, widths, already_padded, **kw: ext.pad(example, widths, already_padded, **kw), self, ext, widths, already_padded=already_padded, **kwargs)

    @staticmethod
    def __concat__(values: tuple, dim: str, **kwargs) -> 'Tensor':
        def example_concat(values, dim, **kw):
            example_values = tuple(v.example_value if isinstance(v, LinTracer) else v for v in values)
            return concat(example_values, dim, **kw)
        return DebugLinTracer._debug_check(LinTracer.__concat__, example_concat, values, dim, **kwargs)

    def _simplify(self):
        return self._debug_check(LinTracer._simplify, lambda example: example, self)


class LinearTraceInProgress(Exception):

    def __init__(self, tracer: Tensor):
        self.tracer = tracer


def trace_linear(f: Callable, *args, auxiliary_args=None, debug_checks=False, **kwargs):
    assert isinstance(auxiliary_args, str) or auxiliary_args is None, f"auxiliary_args must be a comma-separated str but got {auxiliary_args}"
    from ._functional import function_parameters, f_name
    f_params = function_parameters(f)
    aux = set(s.strip() for s in auxiliary_args.split(',') if s.strip()) if isinstance(auxiliary_args, str) else f_params[1:]
    all_args = {**kwargs, **{f_params[i]: v for i, v in enumerate(args)}}
    aux_args = {k: v for k, v in all_args.items() if k in aux}
    trace_args = {k: v for k, v in all_args.items() if k not in aux}
    tree, tensors = disassemble_tree(trace_args, cache=False, attr_type=value_attributes)
    assert len(tensors) == 1, f"Only one input tensor can be traced bot got {tensors}"
    target_backend = backend_for(*tensors)
    # --- Trace function ---
    with NUMPY:
        src = TracerSource(tensors[0].shape, tensors[0].dtype, tuple(trace_args.keys())[0], 0, tensors[0])
        tracer = (DebugLinTracer if debug_checks else LinTracer).create_identity(src)
        x_kwargs = assemble_tree(tree, [tracer] + tensors[1:], attr_type=value_attributes)
        result = f(**x_kwargs, **aux_args)
    out_tree, result_tensors = disassemble_tree(result, cache=False, attr_type=value_attributes)
    assert len(result_tensors) == 1, f"Linear function output must be or contain a single Tensor but got {result}"
    tracer = result_tensors[0]._simplify()
    assert tracer._is_tracer, f"Tracing linear function '{f_name(f)}' failed. Make sure only linear operations are used. Output: {tracer.shape}"
    if debug_checks:
        assert isinstance(tracer, DebugLinTracer), f"Debug checks for linear trace failed. Function might still be correct."
    return out_tree, tracer


def matrix_from_function(f: Callable, *args, auxiliary_args=None,
                         auto_compress=False,
                         target_backend: Backend = None,
                         debug_checks=False,
                         **kwargs) -> Tuple[Tensor, Tensor]:
    """
    Trace a linear function and construct a matrix.
    Depending on the functional form of `f`, the returned matrix may be dense or sparse.

    Args:
        f: Function to trace.
        *args: Arguments for `f`.
        auxiliary_args: Arguments in which the function is not linear.
            These parameters are not traced but passed on as given in `args` and `kwargs`.
        auto_compress: If `True`, returns a compressed matrix if supported by the backend.
        sparsify_batch: If `False`, the matrix will be batched.
            If `True`, will create dual dimensions for the involved batch dimensions.
            This will result in one large matrix instead of a batch of matrices.
        **kwargs: Keyword arguments for `f`.

    Returns:
        matrix: Matrix representing the linear dependency of the output `f` on the input of `f`.
            Input dimensions will be `dual` dimensions of the matrix while output dimensions will be regular.
        bias: Bias for affine functions or zero-vector if the function is purely linear.
    """
    _, tracer = trace_linear(f, *args, auxiliary_args=auxiliary_args, debug_checks=debug_checks, **kwargs)
    return matrix_and_bias_from_tracer(tracer, auto_compress=auto_compress, target_backend=target_backend)


def matrix_and_bias_from_tracer(tracer: Tensor, auto_compress=True, target_backend: Backend = None) -> Tuple[Tensor, Tensor]:
    coo, bias = tracer_to_coo(tracer)
    target_backend = target_backend if target_backend is not None else coo.backend
    matrix = to_format_for_mul(coo, target_backend, auto_compress)
    return matrix, bias
    # sparsify = tracer.shape if sparsify_batch else EMPTY_SHAPE
    # matrix, bias = tracer_to_coo(tracer, sparsify, separate_independent)


def to_format_for_mul(x: Tensor, target_backend: Backend, auto_compress=True) -> Tensor:
    if not is_sparse(x):
        return x
    if auto_compress:
        sparsify_batch = not target_backend.supports(Backend.csr_matrix_batched)
    else:
        sparsify_batch = not target_backend.supports(Backend.sparse_coo_tensor_batched)
    # --- Compress ---
    if x.backend.name == 'torch' and x._values._native.requires_grad:
        auto_compress = False  # PyTorch doesn't support gradient of bincount (used in compression)
    if auto_compress and x.backend.supports(Backend.mul_csr_dense) and target_backend.supports(Backend.mul_csr_dense) and isinstance(x, SparseCoordinateTensor):
        x = x.compress_rows()
    # elif backend.supports(Backend.mul_csc_dense):
    #     return matrix.compress_cols(), tracer._bias
    return x


def leaves_with_offsets(tracer_tree: Tensor, offset: IndexOffset, simplify: bool) -> List[Tuple[Tensor, IndexOffset]]:
    if isinstance(tracer_tree, BlockTensor):
        result = []
        for t, o in tracer_tree._blo:
            result.extend(leaves_with_offsets(t, offset + o, simplify))
        return result
    elif isinstance(tracer_tree, TensorStack):
        result = []
        for i, t in enumerate(tracer_tree._tensors):
            result.extend(leaves_with_offsets(t, offset + {tracer_tree._stack_dim.name: i}, simplify))
        return result
    else:
        if is_sparse(tracer_tree):
            tracer_tree = tracer_tree._with_values(tracer_tree._values._simplify())
            return [(tracer_tree, offset)]  # actual indices will be handled in lin_output_indices
        tracer_tree = tracer_tree._simplify() if simplify else tracer_tree
        return [(tracer_tree, offset)]


def lin_output_indices(x: Tensor, offset: IndexOffset, included_out_dims: Shape) -> Optional[Tensor]:
    """ Assembles an output index tensor for each tracer contained in `tracer_tree`.
    If a single tracer is passed, the output indices simply fill the tracer's shape.
    For composite tensors, takes the relative position into account.
    Output only includes dims along which the dependency is not constant.

    Args:
        x: Input leaf tracer or constant tensor.
        offset: Constant to add to output indices. Dims in `offset` but not in the tensor will be added using the offset value.

    Returns:
        dict: A mapping from tracer instances to corresponding output index tensor matching the tracer's shape.
    """
    single_layer_names = (offset.names & set(included_out_dims.names)) - set(x.shape.names)
    thick_names = set(included_out_dims.names) - single_layer_names
    if not single_layer_names and not thick_names:  # no non-diagonal dependencies
        return None
    with NUMPY:
        if is_sparse(x):
            sp_dims = sparse_dims(x)
            sp_indices = stored_indices(x, instance('sp_entries'))
        else:
            sp_dims = EMPTY_SHAPE
            sp_indices = None
        indices = {}
        for name in thick_names:
            dim = x.shape[name] if name in x.shape else included_out_dims[name]
            if name in sp_dims:
                dim_indices = sp_indices[{channel: name}]
                indices[name] = dim_indices
            else:
                start = offset.by_dim.get(dim.name, 0)
                indices[name] = math.arange(dim, start, start + dim.size)
        for extra in single_layer_names:
            indices[extra] = wrap(offset[extra])
        indices = stack(indices, 'idx:c', expand_values=True)
    return indices
    

def tracer_to_coo(tracer_tree: Tensor) -> Tuple[Tensor, Tensor]:
    tensors_and_offsets = leaves_with_offsets(tracer_tree, NO_OFFSET, simplify=True)
    in_dims = merge_shapes(*[dependent_src_dims(t) for t, _ in tensors_and_offsets])
    out_dims = merge_shapes(*[dependent_out_dims(t, in_dims) for t, _ in tensors_and_offsets])
    offset_names = set.union(*[set(o.by_dim) for _, o in tensors_and_offsets])
    out_dims &= tracer_tree.shape.only(list(offset_names))
    output_indices = [lin_output_indices(x, o, out_dims) for x, o in tensors_and_offsets]
    if len(tensors_and_offsets) == 1 and not in_dims and not out_dims:  # just a scalar multiplication
        tensor = tensors_and_offsets[0][0]
        if tensor._is_tracer:
            assert not (tensor._fac.shape - '_deps')
            return tensor._fac, tensor._bias
        else:
            return wrap(1), tensor  # just a constant
    if not out_dims:  # matrix is a row vector -> make it dense
        matrix = math.zeros(in_dims)
        bias = wrap(0)
        for tensor, _ in tensors_and_offsets:
            if tensor._is_tracer:
                src_indices = tensor._source_indices(included_src_dims=in_dims)
                matrix = scatter(matrix, rename_dims(src_indices, '_deps', instance), rename_dims(tensor._fac, '_deps', instance), 'add', pref_index_dim='idx', outside_handling='undefined')
                bias += tensor._bias  # only scalar bias
            else:  # constant
                bias += tensor
        return rename_dims(matrix, in_dims, dual), bias
    if not in_dims:  # matrix is a column vector -> make it dense
        matrix = math.zeros(out_dims)
        bias = expand(0, out_dims)
        # ToDo bias should pick up non-tracer contributions
        for (tensor, _), out_indices in zip(tensors_and_offsets, output_indices):
            if tensor._is_tracer:
                values = tensor._fac[{'_deps': 0}]  # there can only be 1 input value (in_dims is empty)
                matrix = scatter(matrix, out_indices, values, 'add', pref_index_dim='idx', outside_handling='undefined')
                bias = scatter(bias, out_indices, tensor._bias, 'add', pref_index_dim='idx', outside_handling='undefined')
            else:  # constant
                bias = scatter(bias, out_indices, tensor, 'add', pref_index_dim='idx', outside_handling='undefined')
        return matrix, bias
    else:  # full matrix -> build sparse coo
        with NUMPY:
            bias = math.zeros(out_dims, dtype=tracer_tree.dtype)
        # --- Re-order in_dims to match out_dims if possible to preserve symmetric matrices (e.g. required for CG) ---
        if set(out_dims.names) == set(in_dims.names):
            in_dims = in_dims.only(out_dims.names, reorder=True)
        dual_in = in_dims.as_dual()
        assert dual_in.isdisjoint(out_dims), f"Conflict between input and output dim names. ~Input: {dual_in}, Output: {out_dims}"
        indices = []
        values = []
        for (tensor, _), out_indices in zip(tensors_and_offsets, output_indices):
            if tensor._is_tracer:
                if is_sparse(tensor):
                    src_indices: Tensor = tensor._values._source_indices(included_src_dims=in_dims, as_dual=True, order=in_dims.names)
                    src_indices = expand(src_indices, out_dims.only(tensor.shape) - sparse_dims(tensor))
                    fac = tensor._values._fac
                    fac_nz = is_fac_nonzero(tensor._values)
                    t_bias = tensor._values._bias
                else:
                    src_indices: Tensor = tensor._source_indices(included_src_dims=in_dims, as_dual=True, order=in_dims.names)
                    src_indices = expand(src_indices, out_dims.only(tensor.shape) - src_indices.shape)
                    fac = tensor._fac
                    fac_nz = is_fac_nonzero(tensor)
                    t_bias = tensor._bias
                if out_indices is None:
                    all_indices = src_indices
                else:
                    all_indices = concat([src_indices, out_indices], 'idx', expand_values=True)
                entry_dims = all_indices.shape - 'idx'
                entry_dim = instance(sp_entries=entry_dims.volume)
                entry_idx = pack_dims(all_indices, entry_dims, entry_dim)
                entry_val = pack_dims(expand(fac, entry_dims), entry_dims, entry_dim)
                if not fac_nz.all:  # we know the location of some zeros in fac
                    entry_mask = pack_dims(expand(fac_nz, entry_dims), entry_dims, entry_dim)
                    entry_idx = entry_idx[entry_mask]
                    entry_val = entry_val[entry_mask]
                indices.append(entry_idx)
                values.append(entry_val)
                bias = scatter(bias, out_indices, t_bias, 'add', pref_index_dim='idx', outside_handling='undefined')
            else:  # constant
                bias = scatter(bias, out_indices, tensor, 'add', pref_index_dim='idx', outside_handling='undefined')
        all_indices = concat_tensor(indices, 'sp_entries')
        all_values = concat_tensor(values, 'sp_entries')
        matrix = sparse_tensor(all_indices, all_values, dual_in & out_dims, can_contain_double_entries=True, indices_sorted=False, format='coo', indices_constant=True)
        if isinstance(tracer_tree, LinTracer):
            matrix._prop = TensorProperties(tracer=tracer_tree)
        return matrix, bias


def dependent_src_dims(tracer: Tensor) -> Shape:
    """
    Source dimensions relevant to the linear operation.
    This includes `pattern_dims` as well as dimensions along which only the values vary.
    These dimensions cannot be parallelized trivially with a non-batched matrix.

    Bias dimensions do not require a batched matrix but are subtracted from the right-hand-side vector.
    They are not included unless also relevant to the matrix.
    """
    if not tracer._is_tracer:
        return EMPTY_SHAPE
    if isinstance(tracer, LinTracer):
        dep_names = tracer._indices.shape['idx'].labels[0]
        return tracer._source.shape[list(dep_names)]
    elif is_sparse(tracer):
        values = tracer._values
        return dependent_src_dims(values)
    raise ValueError(tracer)


def dependent_out_dims(tracer: Tensor, included_src_dims: Shape, sparsify=None, matrix_only=True) -> Shape:
    """
    Current dimensions relevant to the linear operation.
    This includes `pattern_dims` as well as dimensions along which only the values vary.
    These dimensions cannot be parallelized trivially with a non-batched matrix.

    Bias dimensions do not require a batched matrix but are subtracted from the right-hand-side vector.
    They are not included unless also relevant to the matrix.
    """
    if not tracer._is_tracer:
        return EMPTY_SHAPE
    if isinstance(tracer, LinTracer):
        out_dims = tracer._var_lin_dims if matrix_only else set(variable_dim_names(tracer))
        dims = tracer.shape.only(out_dims)
        return dims & (included_src_dims.only(tracer.shape) - dims)  # if size changed, prefer from tracer.shape
    elif is_sparse(tracer):
        return sparse_dims(tracer) & dependent_out_dims(tracer._values, included_src_dims).non_instance
    raise ValueError(tracer)


def is_fac_nonzero(tracer: Tensor):
    if isinstance(tracer, LinTracer):
        if tracer._fac.available:
            result = tracer._fac != 0
            return math.convert(result, NUMPY)
        else:
            return tracer._fac_nz
    raise ValueError(tracer)
