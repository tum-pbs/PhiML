import operator
from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Tuple, Dict, Callable, Optional, List, Any, Union

from . import EMPTY_SHAPE
from ._magic_ops import tree_map, all_attributes, tree_broadcast, expand
from ._shape import Shape, merge_shapes, after_gather, shape_stack
from ._tensors import Tensor, _EQUALITY_REDUCE, equality_by_ref, equality_by_shape_and_value
from .. import math
from ..backend import Backend, NUMPY, choose_backend, get_precision, ML_LOGGER
from ..backend._dtype import DType, combine_types, INT32, INT64, BOOL


@dataclass
class Trace:
    name: str
    all_dims: Shape
    distributed: Shape  # dims that could be (or are known to be?) present in the trace input and are distributed
    merge_duplicates: bool = True
    # --- Updated during tracing ---
    all_tracers: List['Tracer'] = field(default_factory=list)
    tracers_by_name: Dict[str, 'Tracer'] = field(default_factory=dict)
    all_ops: List['TracedOp'] = field(default_factory=list)

    def add_input(self, name: str, reference: Tensor):
        assert name not in self.tracers_by_name, f"Duplicate tracers with name {name}"
        tracer = Tracer(self, reference.shape, reference.dtype, {}, None, name)
        self.all_tracers.append(tracer)
        self.tracers_by_name[name] = tracer
        return tracer

    def add_input_d(self, name: str, shape: Shape, dtype: DType):
        assert name not in self.tracers_by_name, f"Duplicate tracers with name {name}"
        tracer = Tracer(self, shape, dtype, {}, None, name)
        self.all_tracers.append(tracer)
        self.tracers_by_name[name] = tracer
        return tracer

    def add_tracers_as_input(self, name: str, tracers: Any):
        def add_single_tracer(x: Tensor):
            return self.add_input(name, x)
        return tree_map(add_single_tracer, tracers, all_attributes, include_non_attrs=False)

    def add_op(self, op_type: str, name: str, args: tuple, req_dims: Shape):
        if self.merge_duplicates:
            for op in self.all_ops:
                if op.name == name and op.op_type == op_type:
                    with equality_by_shape_and_value(0, 0, True):
                        if op.args == args:
                            ML_LOGGER.debug(f"Merging duplicate ops {op.name} in {self.name}")
                            return op
        op = TracedOp(self, op_type, name, args, req_dims)
        self.all_ops.append(op)
        return op

    def add_tracer(self, shape: Shape, dtype: DType, renamed: Dict[str, str], op: 'TracedOp', out_index: int = 0):
        if len(op.outputs) > out_index:
            assert op.outputs[out_index].shape == shape, f"Duplicate op with different output shapes {op.outputs[out_index].shape} vs {shape}"
            return op.outputs[out_index]
        i = 0
        while True:
            name = f"{op.name}[{out_index}]_{i}"
            if name not in self.tracers_by_name:
                break
            i += 1
        tracer = Tracer(self, shape, dtype, renamed, (op, out_index), name)
        self.all_tracers.append(tracer)
        self.tracers_by_name[name] = tracer
        op.outputs.append(tracer)
        return tracer


@dataclass
class TracedOp:
    trace: Trace
    op_type: str  # 'fun', 'op1', 'op2', 'tensor'
    name: str  # function name or operator name, e.g. 'mul'
    args: tuple
    # kwargs: Dict[str, Any]
    req_dims: Shape  # Dimensions that were not acting as batch for the computation of this `Tensor`. This can be any dims present in the sources.
    outputs: List['Tracer'] = field(default_factory=list)

    @cached_property
    def function(self):
        if self.op_type == 'fun':
            return getattr(math, self.name)
        elif self.op_type == 'op2':
            return getattr(operator, self.name)
        elif self.op_type == 'op1':
            return getattr(operator, self.name) if hasattr(operator, self.name) else getattr(math, self.name)
        else:
            assert False, f"Not a function but a {self.op_type} op"

    @cached_property
    def input_tracers(self):
        result = []
        tree_map(lambda t: isinstance(t, Tracer) and result.append(t), self.args, all_attributes, False)
        return result

    def replace_input_tracers(self, new_trace: Trace, replacement: Dict['Tracer', Tensor]):
        args = tree_map(lambda x: replacement[x] if isinstance(x, Tracer) else x, self.args, all_attributes, include_non_attrs=False)
        return new_trace.add_op(self.op_type, self.name, args, self.req_dims)

    @cached_property
    def input_shape(self):
        return merge_shapes(*self.input_tracers, allow_varying_sizes=True)

    @cached_property
    def distributed(self) -> Shape:
        """All dims that should be distributed for this operation."""
        return self.trace.distributed.only(self.input_shape) - self.req_dims

    @cached_property
    def python_imports(self):
        # import {math.__name__} as phiml_math
        if self.op_type == 'fun':
            f = self.function
            return f"import {f.__module__}.{f.__qualname__} as {f.__name__}"
        return ""

    def python_expression(self, tracer_expr: Dict['Tracer', str]):
        if self.op_type == 'op2':
            assert len(self.args) == 2
            op_symbol = {
                'add': '+',
                'sub': '-',
                'mul': '*',
                'truediv': '/',
                'floordiv': '//',
                'pow': '**',
                'mod': '%',
                'matmul': '@',
                'lt': '<',
                'le': '<=',
                'eq': '==',
                'ge': '>=',
                'gt': '>',
                'lshift': '<<',
                'rshift': '>>',
                'and': '&',
                'or': '|',
                'xor': '^',
            }[self.name]
            x, y = [tracer_expr.get(arg, arg._name) if isinstance(arg, Tracer) else str(arg) for arg in self.args]
            return f"({tracer_expr.get(x, str(x))}) {op_symbol} ({tracer_expr.get(y, str(y))})"
        elif self.op_type == 'op1':
            if hasattr(operator, self.name):
                x, = self.args
                op_symbol = {
                    'neg': '-',
                    'inv': '~',
                }.get(self.name)
                if op_symbol is not None:
                    return f"{op_symbol}({tracer_expr[x]})"
                return f"{self.name}({tracer_expr.get(x, str(x))})"
        arg_strs = []
        for i, arg in enumerate(self.args):
            if isinstance(arg, Tracer):
                arg_strs.append(tracer_expr[arg])
            elif isinstance(arg, Shape):
                arg_strs.append(f"'{','.join(arg.names)}'")
            elif isinstance(arg, DType):
                arg_strs.append(f"({arg.kind.__name__}, {arg.bits})")
            else:
                arg_strs.append(str(arg))
        return f"{self.function.__name__}({', '.join(arg_strs)})"

    # ToDo number of using ops (or by tracer?)

    def run(self, data: Dict['Tracer', Tensor]):
        args = tree_map(lambda x: data[x] if isinstance(x, Tracer) else x, self.args, all_attributes, include_non_attrs=False)
        return self.function(*args)

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        return f"{self.name} ({self.op_type})"

    def is_same_op(self, other: 'TracedOp'):
        return self.op_type == other.op_type and self.name == other.name and self.args == other.args


class Tracer(Tensor):

    def __init__(self,
                 trace: Trace,
                 shape: Shape,
                 dtype: DType,
                 renamed: Dict[str, str],
                 op: Optional[Tuple[TracedOp, int]],
                 name: Optional[str] = None):
        """
        Args:
            trace: Trace identifier containing input tensors to the computational graph producing this tensor.
            shape: Tensor shape
            dtype: Tensor data type
            renamed: New dim name -> original name
            op: (op, self-index)
            name: For input tensors
        """
        super().__init__()
        self._trace = trace
        self._shape = shape
        self._dtype = dtype
        self._renamed = renamed  # new_name -> old_name
        self._op = op
        self._name = name

    def _is_tracer(self) -> bool:
        return True

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def dtype(self) -> DType:
        return self._dtype

    def _with_shape_replaced(self, new_shape: Shape):
        renamed = {new_dim: self._renamed[old_dim] for old_dim, new_dim in zip(self._shape.names, new_shape.names)}
        replaced = [o for (o, n) in zip(self._shape, new_shape) if o != n]
        op = self._trace.add_op('fun', 'rename_dims', (self, self._shape, new_shape), merge_shapes(*replaced))
        return self._trace.add_tracer(new_shape, self._dtype, renamed, op)

    @property
    def backend(self) -> Backend:
        return NUMPY

    def __repr__(self):
        return f"{self.shape} // '{self._name}' for {self._trace.name}"

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        if _EQUALITY_REDUCE[-1]['type'] != 'elementwise':  # shape_and_value | ref
            if self is other:
                return True
            if not isinstance(other, Tracer):
                return False
            if self._op is None and other._op is None:
                return self._name == other._name
            if self._op is None or other._op is None:
                return False
            return self._op[1] == other._op[1] and self._op[0].is_same_op(other._op[0])
        return Tensor.__eq__(self, other)

    def _getitem(self, selection: dict):
        req_names = [self._renamed.get(dim, dim) for dim in selection]
        req = self._trace.all_dims.only(req_names)
        new_shape = after_gather(self._shape, selection)
        op = self._trace.add_op('tensor', '[]', (self, selection), req)
        return self._trace.add_tracer(new_shape, self._dtype, self._renamed, op)

    def _unstack(self, dim):
        req_name = self._renamed.get(dim, dim)
        req = self._trace.all_dims[req_name]
        op = self._trace.add_op('fun', 'unstack', (self, dim), req)
        return tuple([self._trace.add_tracer(self._shape - dim, self._dtype, self._renamed, op, i) for i in range(self.shape.get_size(dim))])

    def __cast__(self, dtype: DType):
        op = self._trace.add_op('fun', 'cast', (self, dtype), EMPTY_SHAPE)
        return self._trace.add_tracer(self._shape, dtype, self._renamed, op)

    def _op1(self, native_function):
        raise NotImplementedError("_op1 not supported. Should dispatch to tracer_op1 instead")

    def _op2(self, other, op: Callable, switch_args: bool) -> 'Tensor':
        if isinstance(other, Tensor):
            new_shape = self._shape & other.shape
            dtype = combine_types(self._dtype, other.dtype)
        else:
            new_shape = self._shape
            dtype = combine_types(self._dtype, choose_backend(other).dtype(other))
        if isinstance(other, Tracer):
            renamed = dict(**self._renamed, **other._renamed)
        else:
            renamed = self._renamed
        op = self._trace.add_op('op2', op.__name__, (other, self) if switch_args else (self, other), EMPTY_SHAPE)
        return self._trace.add_tracer(new_shape, dtype, renamed, op)

    def _natives(self):
        raise TraceInProgress(self)

    def _spec_dict(self) -> dict:
        raise TraceInProgress(self)

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **_kwargs) -> 'Tensor':
        tracers = [v for v in values if isinstance(v, Tracer)]
        assert all(t._source == tracers[0]._source for t in tracers)
        new_shape = shape_stack(dim, *[v.shape for v in values])
        dtype = combine_types(*[v.dtype for v in values])
        req_dims = merge_shapes([t.shape for t in tracers])
        op = tracers[0]._trace.add_op('fun', 'stack', (values, dim), req_dims)
        return tracers[0]._trace.add_tracer(new_shape, dtype, tracers[0]._renamed, op)

    def __expand__(self, dims: Shape, **kwargs) -> 'Tracer':
        op = self._trace.add_op('fun', 'expand', (self, dims), EMPTY_SHAPE)
        return Tracer(self._trace, self._shape & dims, self._dtype, self._renamed, (op, 0))

    def native(self, order: Union[str, tuple, list, Shape] = None, force_expand=True):
        assert not self._shape, f"Tracer.native() only supports scalar values but has shape {self._shape}"
        assert not order, f"Tracer.native() does not support transposing or adding dims but got order={order}"
        return self

    def __abs__(self) -> 'Tensor[T]':
        return tracer_op1(self, operator.abs)

    def __round__(self, n=None) -> 'Tensor[int]':
        from ._ops import round_
        return tracer_op1(self, round_)

    def __copy__(self) -> 'Tensor[T]':
        return self

    def __deepcopy__(self, memodict={}) -> 'Tensor[T]':
        return self

    # def __neg__(self) -> 'Tensor[T]':
    #     return tracer_op1(self, operator.neg)

    def __neg__(self):
        op = self._trace.add_op('op1', '-', (self,), EMPTY_SHAPE)
        return self._trace.add_tracer(self._shape, self._dtype, self._renamed, op)

    def __invert__(self) -> 'Tensor[T]':
        return self._op1(lambda t: choose_backend(t).invert(t))


class TraceInProgress(Exception):
    def __init__(self, tracer: Tensor):
        self.tracer = tracer


def tracer_op1(tracer: Tracer, math_fun):
    op = tracer._trace.add_op('op1', math_fun.__name__, (tracer,), EMPTY_SHAPE)
    name = math_fun.__name__
    from ._ops import abs_, to_int64, to_int32, to_float, to_complex, is_finite, is_nan, is_inf, real, imag, stop_gradient, factorial
    old_type = tracer._dtype
    current_float = DType.by_precision(float, get_precision())
    output_type = {
        abs_: old_type,
        to_int32: INT32,
        to_int64: INT64,
        to_complex: DType.by_precision(complex, get_precision()),
        to_float: current_float,
        is_finite: BOOL,
        is_nan: BOOL,
        is_inf: BOOL,
        factorial: old_type,
        real: current_float,
        imag: current_float,
        stop_gradient: old_type,
    }.get(name, current_float)
    return tracer._trace.add_tracer(tracer._shape, output_type, tracer._renamed, op)


def tracer_reduce(tracer: Tracer, dims: Shape, op_name: str, fun):
    distributed = tracer._trace.distributed.only(dims)
    local = dims - distributed
    if distributed and local:
        return tracer_reduce(tracer_reduce(tracer, local, op_name, fun), distributed, op_name, fun)
    new_shape = tracer._shape - dims
    renamed = {k: v for k, v in tracer._renamed.items() if k not in dims}
    req_names = [tracer._renamed.get(dim, dim) for dim in dims.names]
    req = tracer._trace.all_dims.only(req_names)  # ToDo actually, we want a clean slate for the next steps...
    op = tracer._trace.add_op('fun', op_name, (tracer, dims), req)
    return tracer._trace.add_tracer(new_shape, tracer._dtype, renamed, op)


@tree_broadcast(all_attributes, treat_shapes_as_leaf=True, include_non_attrs=False)
def to_tracers(obj: Any):
    if isinstance(obj, Tracer):
        return obj
    trace = Trace('<default>', EMPTY_SHAPE, EMPTY_SHAPE)
    if isinstance(obj, Tensor):
        return Tracer(trace, obj.shape, obj.dtype, {}, None)
    if isinstance(obj, Shape):
        return Tracer(trace, obj, DType.as_dtype(float), {}, None)
    return obj


def expand_tracers(tracers, dims: Shape):
    return tree_map(lambda x: expand(x, dims), tracers, all_attributes, include_non_attrs=False)