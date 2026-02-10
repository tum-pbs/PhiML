import operator
import warnings
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Dict, Set, Tuple, Union, Optional, Sequence, List

from ..backend import NUMPY, Backend
from ..backend import get_precision
from ..backend._dtype import DType, combine_types
from . import _ops as math
from ._magic_ops import stack, expand, rename_dims, unpack_dim, value_attributes, ccat, pack_dims, concat
from ._ops import backend_for, concat_tensor, scatter
from ._shape import Shape, merge_shapes, instance, EMPTY_SHAPE, dual, channel, non_batch, non_channel, DEBUG_CHECKS, \
    after_gather, concat_shapes_, batch
from ._sparse import SparseCoordinateTensor, is_sparse, sparse_dims, sparse_tensor, stored_indices, stored_values, add_sparse_batch_dim
from ._tensors import Tensor, wrap, TensorStack, discard_constant_dims, variable_shape, Dense, BlockTensor, NO_OFFSET, IndexOffset, variable_dim_names
from ._tree import disassemble_tree, assemble_tree
from ._nd import vec
from .extrapolation import Extrapolation

TracerSource = namedtuple('TracerSource', ['shape', 'dtype', 'name', 'index'])


@dataclass(frozen=True, eq=False, unsafe_hash=False, repr=False)
class LinTracer(Tensor):
    """ Uniform `Tensor` where each output value depends on a fixed number of input values.
    These dependencies are listed in `_fac` along `_deps:b`.
    The shape of LinTracer is equal to the `_bias` shape.
    """
    _source: TracerSource
    _indices: Tensor
    """ Shape compatible with self.shape.
    Special dims:
        * channel dim 'idx': contains only relevant source dims, other src dim dependence is constant.
        * batch dim `_deps': contributions from multiple input indices to be summed
    """
    _fac: Tensor
    """ multiplication factors: sum(mul * src[indices], '_deps'). Can have fewer dims than indices. Shape compatible with self.shape. """
    _bias: Tensor
    """ Shape equal to `self.shape`, dtype equal to `self.dtype`. Can contain additional expanded dims along which values (not dependencies) are constant """

    @classmethod
    def create_identity(cls, src: TracerSource):
        indices = math.zeros(batch(_deps=1), channel(idx=''))
        fac = math.ones(batch(_deps=1))
        bias = math.zeros(src.shape, dtype=src.dtype)
        return cls(src, indices, fac, bias)

    def _source_indices(self, included_out_dims: Shape = EMPTY_SHAPE, included_src_dims: Shape = EMPTY_SHAPE, as_dual=False):
        """
        Args:
            included_dims: Dim names in `self.shape` that should be part of the result's shape even if the dependency is constant along them.
        """
        extend = []  # index components to add because of included_dims
        constant_dims = []
        with NUMPY:
            for dim in included_src_dims - self._var_src_names:  # dims that are not yet in self._indices, constant dependence (diagonal)
                assert dim in self.shape, f"Cannot add source dim {dim} after it has been sliced off (not in self.shape)"
                extend.append(vec('idx', **{dim.name: math.arange(self._bias.shape[dim.name])}))
            for dim in included_out_dims:  # dims that are not yet in self._indices, constant dependence (diagonal)
                raise NotImplementedError
                if dim in self.shape:
                    src_name = dim.name
                    assert self._source.shape.get_size(src_name) == self._bias.shape.get_size(dim.name), f"Dim size has changed from {self._source.shape.get_size(src_name)} to {self._bias.shape.get_size(dim.name)} despite not being included in indices."
                    extend.append(vec('idx', **{src_name: math.arange(self._bias.shape[dim.name])}))
                else:
                    constant_dims.append(dim)
        as_primal = concat([self._indices, *extend], 'idx', expand_values=True)
        as_primal = expand(as_primal, *constant_dims)
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
        bias = self._bias._with_shape_replaced(new_shape)
        return LinTracer(self._source, indices, fac, bias)

    def _getitem(self, selection: dict) -> 'Tensor':
        new_dims = self._source.shape.only(tuple(selection)) - self._indices.shape
        indices = self._source_indices(included_src_dims=new_dims)[selection]
        fac = self._fac[selection]
        bias = self._bias[selection]
        return LinTracer(self._source, indices, fac, bias)

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
                idx1 = t1._source_indices(included_src_dims=src_dims)
                idx2 = t2._source_indices(included_src_dims=src_dims)
                indices = concat([idx1, idx2], '_deps', expand_values=True)
                fac = concat([t1._fac, t2._fac], '_deps', expand_values=True)
                bias = t1._bias + t2._bias
                return LinTracer(self._source, indices, fac, bias)
            return BlockTensor(t1.shape & t2.shape, [(t1, NO_OFFSET), (t2, NO_OFFSET)], operator.add)
        else:  # op with constant
            other = self._tensor(other)
            bias = op(self._bias, other)
            if op in {operator.mul, operator.truediv}:
                fac = op(self._fac, other)
                return LinTracer(self._source, self._indices, fac, bias)
            elif op in {operator.add, operator.sub}:
                return LinTracer(self._source, self._indices, self._fac, bias)
            else:
                raise ValueError(f"Unsupported operation encountered while tracing linear function: {op}")

    def _op1(self, native_function, op_name: str) -> Tensor:
        # __neg__ and __cast__ implemented below
        if native_function.__name__ == 'isfinite':
            return expand(math.is_finite(self._fac), self.shape)
        elif op_name in {'cast', 'to_float', 'to_int32', 'to_int64', 'to_complex'}:
            raise AssertionError("cast called via _op1. Should be __cast__ instead")
        else:
            raise NotImplementedError('Only linear operations are supported')

    def __neg__(self):
        return LinTracer(self._source, self._indices, -self._fac, -self._bias)

    def __cast__(self, dtype: DType) -> 'Tensor':
        if self.dtype == dtype:
            return self
        if self._source.dtype & dtype == self.dtype:  # cannot down-cast
            warnings.warn(f"Cannot cast linear tracer of type {self.dtype} to {dtype} because its input has type {self._source.dtype}", RuntimeWarning)
            return self
        fac = math.cast(self._fac, dtype)
        bias = math.cast(self._bias, dtype)
        return LinTracer(self._source, self._indices, fac, bias)

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
        bias = math.sum_(self._bias, dims)
        return LinTracer(self._source, indices, fac, bias)

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
        return LinTracer(self._source, idx, fac, bias)

    def _matmul(self, self_dims: Shape, matrix: Tensor, matrix_dims: Shape) -> Tensor:
        shape = matrix.shape.without(matrix_dims) & self._shape.without(self_dims)
        raise NotImplementedError
        if self_dims not in self._diag.shape:  # self is constant along self_dims
            matrix = math.sum_(matrix, matrix_dims)
        diag = matrix * self._diag
        diag = rename_dims(diag, matrix_dims, rename_dims(self_dims, [*self._renamed.keys()], [*self._renamed.values()]).as_dual())
        renamed = {n: o for n, o in self._renamed.items() if n not in self_dims}
        return GatherLinTracer(self._source, diag, self._bias, shape, self._selection, renamed)

    def _scatter(self, base: Tensor, indices: Tensor) -> Tensor:
        # assume base is 0
        values = self.__pack_dims__(non_batch(self), instance('entries'), None)
        return sparse_tensor(indices, values, base.shape & batch(self))

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **_kwargs) -> 'Tensor':
        if any(not isinstance(v, LinTracer) for v in values):
            return NotImplemented
        if len(values) == 1:
            return values[0].__expand__(dim)
        src_dims = merge_shapes(*[dependent_src_dims(t) for t in values])
        indices = [t._source_indices(included_src_dims=src_dims) for t in values if isinstance(t, LinTracer)]
        indices = stack(indices, dim, expand_values=True)
        fac = stack([t._fac for t in values], dim, expand_values=True)
        bias = stack([t._bias for t in values], dim)
        return LinTracer(values[0]._source, indices, fac, bias)

    def __expand__(self, dims: Shape, **kwargs) -> 'Tensor':
        return LinTracer(self._source, self._indices, self._fac, expand(self._bias, dims))

    def _pad(self, ext: Extrapolation, widths, already_padded, **kwargs):
        assert not already_padded
        no_bias = ext - ext  # ToDo for constant extrapolation, return a composite tensor, so we don't have to filter out zero-values later (which may be impossible when jit-compiling)
        indices = self._source_indices(included_src_dims=self._source.shape.only(tuple(widths)))
        indices = no_bias.pad(indices, widths)
        fac = no_bias.pad(expand(self._fac, self._source.shape.only(tuple(widths))), widths)
        bias = ext.pad(self._bias, widths)
        return LinTracer(self._source, indices, fac, bias)

    @staticmethod
    def __concat__(values: tuple, dim: str, **kwargs) -> 'Tensor':
        full_size = sum([t_.shape.get_size(dim) for t_ in values])
        shape = merge_shapes([t.shape.with_dim_size(dim, full_size) for t in values])
        if any(not isinstance(t, LinTracer) for t in values):
            raise NotImplementedError  # BlockTensor
        # --- Concat only LinTracers ---
        src_dims = merge_shapes(*[dependent_src_dims(t) for t in values]).with_dim_size(dim, None) & shape[dim].with_dim_size(dim, None)
        indices = [t._source_indices(included_src_dims=src_dims) for t in values if isinstance(t, LinTracer)]
        indices = concat(indices, dim, expand_values=True)
        fac = concat([expand(t._fac, t.shape[dim]) for t in values], dim, expand_values=True)
        bias = concat([t._bias for t in values], dim)
        return LinTracer(values[0]._source, indices, fac, bias)

    def min_rank_deficiency(self) -> Tensor:
        raise NotImplementedError
        trimming_dict = {}
        for dim in pattern_dim_names(self):
            shifts = [shift[dim] for shift in self.val]
            lo = -min(shifts)
            hi = -max(shifts) or None
            trimming_dict[dim] = slice(lo, hi)
        trimmed_vals = [v[trimming_dict] for v in self.val.values()]
        if all(v.available for v in trimmed_vals):
            stencil_sum = sum(trimmed_vals)
            stencil_abs = sum([abs(v) for v in trimmed_vals])
            eps = {16: 1e-2, 32: 1e-5, 64: 1e-10}[get_precision()]
            balanced_stencil = math.close(0, stencil_sum, rel_tolerance=0, abs_tolerance=eps * math.mean(stencil_abs), reduce=pattern_dim_names(self))
        else:
            balanced_stencil = True  # cannot be determined here because values can vary. Assume could be rank-deficient to print warning
        deficiency = 0
        for shift, nonzero in self._nz_edge.items():
            if shift and nonzero:
                deficiency += 1
        return math.where(balanced_stencil, deficiency, 0)

    def __repr__(self):
        return f"{self.__class__.__name__} {self._bias.shape}"


class LinearTraceInProgress(Exception):

    def __init__(self, tracer: Tensor):
        self.tracer = tracer


def trace_linear(f: Callable, *args, auxiliary_args=None, **kwargs):
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
        src = TracerSource(tensors[0].shape, tensors[0].dtype, tuple(trace_args.keys())[0], 0)
        tracer = LinTracer.create_identity(src)
        x_kwargs = assemble_tree(tree, [tracer] + tensors[1:], attr_type=value_attributes)
        result = f(**x_kwargs, **aux_args)
    out_tree, result_tensors = disassemble_tree(result, cache=False, attr_type=value_attributes)
    assert len(result_tensors) == 1, f"Linear function output must be or contain a single Tensor but got {result}"
    tracer = result_tensors[0]._simplify()
    assert tracer._is_tracer, f"Tracing linear function '{f_name(f)}' failed. Make sure only linear operations are used. Output: {tracer.shape}"
    return out_tree, tracer


def matrix_from_function(f: Callable, *args, auxiliary_args=None,
                         auto_compress=True,
                         target_backend: Backend = None,
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
    _, tracer = trace_linear(f, *args, auxiliary_args=auxiliary_args, **kwargs)
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


def leaves_with_offsets(tracer_tree: Tensor, offset: IndexOffset) -> List[Tuple[Tensor, IndexOffset]]:
    if isinstance(tracer_tree, BlockTensor):
        result = []
        for t, o in tracer_tree._blo:
            result.extend(leaves_with_offsets(t, offset + o))
        return result
    elif isinstance(tracer_tree, TensorStack):
        result = []
        for i, t in enumerate(tracer_tree._tensors):
            result.extend(leaves_with_offsets(t, offset + {tracer_tree._stack_dim.name: i}))
        return result
    elif is_sparse(tracer_tree):
        raise NotImplementedError
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
        indices = {}
        for name in thick_names:
            dim = x.shape[name] if name in x.shape else included_out_dims[name]
            start = offset.by_dim.get(dim.name, 0)
            indices[dim.name] = math.arange(dim, start, start + dim.size)
        for extra in single_layer_names:
            indices[extra] = wrap(offset[extra])
        indices = stack(indices, 'idx:c', expand_values=True)
    return indices
    

def tracer_to_coo(tracer_tree: Tensor) -> Tuple[Tensor, Tensor]:
    tensors_and_offsets = leaves_with_offsets(tracer_tree, NO_OFFSET)
    in_dims = merge_shapes(*[dependent_src_dims(t) for t, _ in tensors_and_offsets])
    out_dims = merge_shapes(*[dependent_out_dims(t, in_dims) for t, _ in tensors_and_offsets])
    offset_names = set.union(*[set(o.by_dim) for _, o in tensors_and_offsets])
    out_dims &= tracer_tree.shape.only(list(offset_names))
    output_indices = [lin_output_indices(x, o, out_dims) for x, o in tensors_and_offsets]
    dual_in = in_dims.as_dual()
    assert dual_in.isdisjoint(out_dims), f"Conflict between input and output dim names. ~Input: {dual_in}, Output: {out_dims}"
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
                matrix = scatter(matrix, rename_dims(src_indices, '_deps', instance), rename_dims(tensor._fac, '_deps', instance), 'add', pref_index_dim='idx')
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
                matrix = scatter(matrix, out_indices, values, 'add', pref_index_dim='idx')
                bias = scatter(bias, out_indices, tensor._bias, 'add', pref_index_dim='idx')
            else:  # constant
                bias = scatter(bias, out_indices, tensor, 'add', pref_index_dim='idx')
        return matrix, bias
    else:  # full matrix -> build sparse coo
        bias = expand(0, out_dims)
        indices = []
        values = []
        for (tensor, _), out_indices in zip(tensors_and_offsets, output_indices):
            if tensor._is_tracer:
                src_indices: Tensor = tensor._source_indices(included_src_dims=in_dims, as_dual=True)
                src_indices = expand(src_indices, out_dims.only(tensor.shape) - src_indices.shape)
                if out_indices is None:
                    all_indices = src_indices
                else:
                    all_indices = concat([src_indices, out_indices], 'idx', expand_values=True)
                entry_dims = all_indices.shape - 'idx'
                entry_dim = instance(entries=entry_dims.volume)
                entry_idx = pack_dims(expand(all_indices, entry_dims), entry_dims, entry_dim)
                entry_val = pack_dims(expand(tensor._fac, entry_dims), entry_dims, entry_dim)
                indices.append(entry_idx)
                values.append(entry_val)
                bias = scatter(bias, out_indices, tensor._bias, 'add', pref_index_dim='idx')
            else:  # constant
                bias = scatter(bias, out_indices, tensor, 'add', pref_index_dim='idx')
        indices = concat_tensor(indices, 'entries')
        values = concat_tensor(values, 'entries')
        matrix = sparse_tensor(indices, values, dual_in & out_dims, can_contain_double_entries=True, indices_sorted=False, format='coo', indices_constant=True)
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
        dep_names = set(tracer._indices.shape['idx'].labels[0])
        return tracer._source.shape[list(dep_names)]
    raise ValueError(tracer)


def dependent_out_dims(tracer: Tensor, included_src_dims: Shape, sparsify=None) -> Shape:
    """
    Current dimensions relevant to the linear operation.
    This includes `pattern_dims` as well as dimensions along which only the values vary.
    These dimensions cannot be parallelized trivially with a non-batched matrix.

    Bias dimensions do not require a batched matrix but are subtracted from the right-hand-side vector.
    They are not included unless also relevant to the matrix.
    """
    if isinstance(tracer, LinTracer):
        out_dims = set(variable_dim_names(tracer))
        dims = tracer.shape.only(out_dims)
        return dims & (included_src_dims.only(tracer.shape) - dims)  # if size changed, prefer from tracer.shape
    raise ValueError(tracer)
