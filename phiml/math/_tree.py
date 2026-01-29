import copy
import dataclasses
import operator
import os.path
import warnings
from functools import partial
from numbers import Number
from typing import Tuple, Callable, List
from typing import Union, TypeVar, Sequence, Any, Dict

import numpy
import numpy as np

from ._shape import Shape, CHANNEL_DIM, EMPTY_SHAPE, parse_dim_order, shape_stack, channel, batch, spatial, shape, DEBUG_CHECKS, parse_shape_spec, after_gather, concat_shapes_, \
    Dim, PureShape, SHAPE_TYPES, prepare_renaming_gather, to_spec_str, split_top_level_comma, from_spec_str
from ._tensors import Tensor, wrap, BROADCAST_FORMATTER, T, Dense, unserialize_spec, assemble_tensors, serialize_spec, _EQUALITY_REDUCE, TensorStack
from .magic import PhiTreeNode, slicing_dict
from .magic import Shapable
from ..backend import NoBackendFound, choose_backend, OBJECTS, NUMPY
from ..backend._backend import get_operator
from ..backend._dtype import DType, BOOL, INT64, OBJECT

PhiTreeNodeType = TypeVar('PhiTreeNodeType')  # Defined in phiml.math.magic: tuple, list, dict, custom


class Layout(Tensor):
    """
    Tensor representation of a PyTree consisting of only lists, tuples and leaves.
    Leaves can be any Python object or primitive, including tuples and lists.
    The PyTree may be deeper but only the outer `shape.rank` levels are represented as a tensor.
    """

    def __init__(self, obj, stack_dim: Shape):
        super().__init__()
        self._obj = obj
        obj_shapes = Layout._recursive_get_shapes(obj, stack_dim)
        self._shape = shape_stack(stack_dim, *obj_shapes, stack_dim_first=True)
        self._stack_dim = stack_dim
        if DEBUG_CHECKS:
            if self._stack_dim:
                assert stack_dim == self._shape[:stack_dim.rank]
            elif isinstance(obj, Shapable) and obj is not None:
                warnings.warn(f"Empty stack_dim for Layout with value {obj}")

    @staticmethod
    def _recursive_get_shapes(obj, s: Shape) -> Tuple[Shape, ...]:
        if not s:
            return shape(obj, allow_unshaped=True),
        elif isinstance(obj, (tuple, list)):
            return sum([Layout._recursive_get_shapes(o, after_gather(s, {s.names[0]: i})) for i, o in enumerate(obj)], ())
        elif isinstance(obj, dict):
            return sum([Layout._recursive_get_shapes(v, after_gather(s, {s.names[0]: i})) for i, (k, v) in enumerate(obj.items())], ())
        obj_shape = shape(obj, allow_unshaped=True)
        return (obj_shape,) * s.volume

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def dtype(self) -> DType:
        if isinstance(self._obj, bool):
            return BOOL
        if isinstance(self._obj, int):
            return INT64
        elif isinstance(self._obj, (float, complex)):
            return DType.by_precision(type(self._obj), precision=64)
        else:
            return OBJECT

    @property
    def default_backend(self):
        return None

    def item(self):
        return self._obj

    def native(self, order: Union[str, tuple, list, Shape] = None, force_expand=True, to_numpy=False):
        if order is not None:
            order = parse_dim_order(order)
            assert order == self._stack_dim.names, "Layout.native() does not allow for changing the dimension order"
        native = self._obj
        return numpy.asarray(native) if to_numpy else native

    def _cached(self):
        from ._ops import cached
        return Layout(cached(self._obj), self._stack_dim)

    def __getitem__(self, item):
        if item is None:
            return self
        item = slicing_dict(self, item)
        return self._getitem(item)

    def _getitem(self, selection: dict) -> 'Tensor':
        selection_list = []
        for dim in self._stack_dim:
            sel, new_dim = prepare_renaming_gather(self.shape, dim, selection.get(dim.name, None))
            selection_list.append(sel)
        native = self._getitem_recursive(self._obj, tuple(selection_list), selection)
        return Layout.wrap(native, after_gather(self._stack_dim, selection))

    @staticmethod
    def wrap(native, stack_dim: Shape):
        if isinstance(native, Tensor):
            return native
        if isinstance(native, Shapable) and native is not None and not isinstance(native, (tuple, list, dict)):
            # maybe allow class to configure whether to be unpacked
            return native
        if isinstance(native, (bool, Number)):
            return wrap(native)
        return Layout(native, stack_dim)

    def __repr__(self):
        return repr(self._obj)

    def __format__(self, format_spec):
        if BROADCAST_FORMATTER.values is not None:
            return BROADCAST_FORMATTER.register_formatted(self, format_spec)
        return repr(self._obj)

    def _unstack(self, dimension: str):
        if dimension == self._stack_dim.names[0]:
            native = tuple(self._obj.values()) if isinstance(self._obj, dict) else self._obj
            inner_stack_dim = self._stack_dim[1:]
            return tuple([Layout.wrap(n, inner_stack_dim) for n in native])
        else:
            raise NotImplementedError()

    @staticmethod
    def _getitem_recursive(native, selection: tuple, sel_dict: dict):
        if not selection:
            return native
        native = tuple(native.values()) if isinstance(native, dict) else native
        if len(selection) == 1:
            sel = selection[0]
            if sel is not None:
                if isinstance(sel, (tuple, list)):
                    assert isinstance(native, (tuple, list))
                    native = [native[i] for i in sel]
                elif isinstance(sel, (int, slice)):
                    native = native[sel]
            return slice_(native, sel_dict)
        else:
            if selection[0] is None:
                return type(native)([Layout._getitem_recursive(n, selection[1:], sel_dict) for n in native])
            if isinstance(selection[0], int):
                return Layout._getitem_recursive(native[selection[0]], selection[1:], sel_dict)
            elif isinstance(selection[0], slice):
                subset = native[selection[0]]
                return type(subset)([Layout._getitem_recursive(n, selection[1:], sel_dict) for n in subset])
            else:
                raise ValueError(f"Illegal selection: {selection}")

    def _as_list(self):
        return self._as_list_recursive(self._obj, self._stack_dim.rank, [])

    @staticmethod
    def _as_list_recursive(native, dims: int, result: list):
        if dims == 0:
            result.append(native)
        else:
            native = tuple(native.values()) if isinstance(native, dict) else native
            for n in native:
                Layout._as_list_recursive(n, dims - 1, result)
        return result

    @property
    def _is_tracer(self) -> bool:
        return False

    def __bool__(self):
        assert self.rank == 0, f"Cannot convert tensor with non-empty shape {self.shape} to bool. Use tensor.any or tensor.all instead."
        return bool(self._obj)

    def __stack__(self, values: tuple, dim: Shape, **kwargs) -> 'Layout':
        obj = [v.native(self._stack_dim) for v in values]
        new_stack_dim = shape_stack(dim, *[v._stack_dim for v in values])
        return Layout(obj, new_stack_dim)

    @staticmethod
    def __concat__(values: tuple, dim: str, **kwargs) -> 'Shapable':
        return NotImplemented

    def __flatten__(self, flat_dim: Shape, flatten_batch: bool):
        return NotImplemented

    def __expand__(self, dims: Shape, **kwargs) -> 'Tensor':
        new_stack_dims = dims.without(self._stack_dim)
        if not new_stack_dims:
            return self
        obj = self._obj
        for dim in reversed(new_stack_dims):
            assert isinstance(dim.size, int), "Can only expand layouts by integer-sized dimensions"
            obj = [obj] * dim.size
        return Layout(obj, new_stack_dims + self._stack_dim)

    def __replace_dims__(self, dims: Tuple[str, ...], new_dims: Shape, **kwargs) -> 'Tensor':
        new_stack_dim = self._stack_dim.replace(dims, new_dims)
        from ._magic_ops import rename_dims
        def inner_replace(obj):
            return rename_dims(obj, dims, new_dims, **kwargs)
        obj = self._recursive_op1(self._obj, self._stack_dim, inner_replace)
        return Layout(obj, new_stack_dim)

    def __pack_dims__(self, dims: Shape, packed_dim: Shape, pos: Union[int, None], **kwargs) -> 'Layout':
        if dims.names == self._stack_dim.names:
            native = self._as_list()
            return Layout(native, packed_dim.with_size(len(native)))
        else:
            obj = []
            for i in dims.meshgrid():
                obj.append(self[i].native())
            return Layout(obj, packed_dim.with_size(dims.volume) + (self._stack_dim - dims))

    def __unpack_dim__(self, dim: str, unpacked_dims: Shape, **kwargs) -> 'Layout':
        return NotImplemented

    def __cast__(self, dtype: DType):
        obj = self._recursive_cast(self._obj, self._stack_dim, dtype)
        return Layout(obj, self._stack_dim)

    def __copy__(self):
        return Layout(self._obj, self._stack_dim)

    def __iter__(self):
        if self.rank == 1:
            return iter(self._obj)
        elif self.rank == 0:
            return iter([self._obj])
        else:
            return iter(self._as_list())

    def __eq__(self, other):
        if _EQUALITY_REDUCE[-1]['type'] != 'elementwise':
            return Tensor.__eq__(self, other)
        return self._op2(other, operator.eq, False)

    def __ne__(self, other):
        if _EQUALITY_REDUCE[-1]['type'] != 'elementwise':
            return Tensor.__ne__(self, other)
        return self._op2(other, operator.ne, False)

    def _assert_close(self, other: Tensor, rel_tolerance: float, abs_tolerance: float, msg: str, verbose: bool):
        from ._ops import assert_close
        inner_test = lambda x, y: assert_close(x, y, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance, msg=msg, verbose=verbose)
        return self._op2(other, inner_test, False)

    def _op2(self, other, op: Callable, switch_args: bool) -> 'Tensor':
        obj = self._recursive_op2(self._obj, self._stack_dim, other, op)
        new_stack = self._stack_dim + (other._stack_dim - self._stack_dim) if isinstance(other, Layout) else self._stack_dim
        return Layout(obj, new_stack)

    @staticmethod
    def _recursive_op2(obj, shape: Shape, other: Tensor, operator: Callable):
        if shape:
            dim = shape.names[0]
            if isinstance(other, Tensor) and dim in other.shape:
                assert other.shape.get_size(dim) == len(obj), f"Shape mismatch during {operator.__name__}: '{dim}' has size {len(obj)} on layout but {other.shape.get_size(dim)} on other tensor."
                others = [other[{dim: i}] for i in range(len(obj))]
            else:
                others = [other] * len(obj)
            if isinstance(obj, (tuple, list)):
                return type(obj)([Layout._recursive_op2(i, shape[1:], o, operator) for i, o in zip(obj, others)])
            elif isinstance(obj, dict):
                return {k: Layout._recursive_op2(v, shape[1:], o, operator) for (k, v), o in zip(obj.items(), others)}
        else:  # leaf
            if isinstance(other, Layout) and not other.shape:
                return operator(obj, other.native())
            if isinstance(other, Tensor):
                return operator(obj, other)
            else:
                if obj is None or other is None:
                    return operator(obj, other)
                native_function = get_operator(operator, choose_backend(obj, other))
                return native_function(obj, other)

    def _op1(self, native_function, op_name: str):
        return Layout(self._recursive_op1(self._obj, self._stack_dim, native_function), self._stack_dim)

    @staticmethod
    def _recursive_op1(obj, shape: Shape, native_function):
        if shape:
            if isinstance(obj, (tuple, list)):
                return type(obj)([Layout._recursive_op1(i, shape[1:], native_function) for i in obj])
            elif isinstance(obj, dict):
                return {k: Layout._recursive_op1(v, shape[1:], native_function) for k, v in obj.items()}
            raise ValueError(obj)
        else:
            return native_function(obj)

    @staticmethod
    def _recursive_cast(obj, shape: Shape, dtype: DType):
        if shape:
            if isinstance(obj, (tuple, list)):
                return type(obj)([Layout._recursive_cast(i, shape[1:], dtype) for i in obj])
            elif isinstance(obj, dict):
                return {k: Layout._recursive_cast(v, shape[1:], dtype) for k, v in obj.items()}
            elif isinstance(obj, Tensor):
                assert obj.shape == shape
                from ._ops import cast
                return cast(obj, dtype)
            raise ValueError(obj)
        elif isinstance(obj, Tensor):
            from ._magic_ops import cast
            return cast(obj, dtype)
        else:
            return dtype.kind(obj)


def layout(objects: Union[Sequence[T], T], *shape: Union[Shape, str]) -> Tensor[T]:
    """
    Wraps a Python tree in a `Tensor`, allowing elements to be accessed via dimensions.
    A python tree is a structure of nested `tuple`, `list`, `dict` and *leaf* objects where leaves can be any Python object.

    All keys of `dict` containers must be of type `str`.
    The keys are automatically assigned as labels along that dimension unless conflicting with other elements.

    Strings may also be used as containers.

    Example:
    >>> t = layout({'a': 'text', 'b': [0, 1]}, channel('dict,inner'))
    >>> t.inner[1].dict['a'].native()
    'e'

    See Also:
        `tensor()`, `wrap()`.

    Args:
        objects: PyTree of `list` or `tuple`.
        *shape: Tensor dimensions

    Returns:
        `Tensor`.
        Calling `Tensor.native()` on the returned tensor will return `objects`.
    """
    shape = [parse_shape_spec(s) if isinstance(s, str) else s for s in shape]
    assert all(isinstance(s, SHAPE_TYPES) for s in shape), f"shape needs to be one or multiple Shape instances but got {shape}"
    shape = EMPTY_SHAPE if len(shape) == 0 else concat_shapes_(*shape)
    if isinstance(objects, Layout):
        assert objects.shape == shape
        return objects

    if not shape.well_defined:

        def recursive_determine_shape(native, shape: Shape):
            if not shape:
                return shape
            if isinstance(native, dict):
                assert all([isinstance(k, str) for k in native.keys()]), f"All dict keys in PyTrees must be str but got {tuple(native.keys())}"
                shape = shape.replace(shape[0], shape[0].with_size(tuple(native.keys())))
            if shape.rank == 1:
                return shape.with_sizes((len(native),))
            inner_shape = shape[1:]
            if isinstance(native, (tuple, list)):
                inner_shapes = [recursive_determine_shape(n, inner_shape) for n in native]
            elif isinstance(native, dict):
                inner_shapes = [recursive_determine_shape(n, inner_shape) for n in native.values()]
            else:
                raise ValueError(native)
            return shape_stack(shape[0], *inner_shapes)

        shape = recursive_determine_shape(objects, shape)

    return Layout(objects, shape)


def is_composite(x: Any) -> bool:
    """
    Args:
        x: Object to check.

    Returns:
        `True` if `x` is a composite type / container, e.g. a dataclass or pytree.
        Sparse tensors are treated as non-composite.
    """
    if x is None:
        return False
    elif isinstance(x, Layout):
        return True
    elif isinstance(x, Tensor):
        return False
    elif dataclasses.is_dataclass(x):
        return True
    elif isinstance(x, (tuple, list, dict)):
        return True
    try:
        backend = choose_backend(x)
        return not backend.is_tensor(x)
    except NoBackendFound as err:
        raise ValueError(x) from err


def object_dims(value):
    """For composite types, returns the dims along which objects are arranged, excluding numeric tensor dims."""
    if isinstance(value, Layout):
        return value._stack_dim
    return EMPTY_SHAPE


def replace(obj: PhiTreeNodeType, **updates) -> PhiTreeNodeType:
    """
    Creates a copy of the given `phiml.math.magic.PhiTreeNode` with updated values as specified in `updates`.

    If `obj` overrides `__with_attrs__`, the copy will be created via that specific implementation.
    Otherwise, the `copy` module and `setattr` will be used.

    Args:
        obj: `phiml.math.magic.PhiTreeNode`
        **updates: Values to be replaced.

    Returns:
        Copy of `obj` with updated values.
    """
    if isinstance(obj, (Number, bool)):
        return obj
    elif hasattr(obj, '__with_attrs__'):
        result = obj.__with_attrs__(**updates)
        if result is not NotImplemented:
            return result
    if dataclasses.is_dataclass(obj):
        return dataclasses.replace(obj, **updates)
    else:
        cpy = copy.copy(obj)
        for attr, value in updates.items():
            setattr(cpy, attr, value)
        return cpy


copy_with = replace


def variable_attributes(obj) -> Tuple[str, ...]:
    if hasattr(obj, '__variable_attrs__'):
        result = obj.__variable_attrs__()
        assert isinstance(result, tuple), f"__variable_attrs__ must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
        return result
    elif hasattr(obj, '__all_attrs__'):
        result = obj.__all_attrs__()
        assert isinstance(result, tuple), f"__all_attrs__ must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
        return result
    elif hasattr(obj, '__value_attrs__'):
        result = obj.__value_attrs__()
        assert isinstance(result, tuple), f"__value_attrs__ must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
        return result
    elif dataclasses.is_dataclass(obj):
        if hasattr(obj, 'variable_attrs'):
            result = obj.variable_attrs
            assert isinstance(result, tuple), f"dataclass.variable_attrs must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
            return result
        return all_attributes(obj)
    else:
        raise ValueError(f"Not a PhiTreeNode: {type(obj).__name__}")


def value_attributes(obj) -> Tuple[str, ...]:
    if hasattr(obj, '__value_attrs__'):
        result = obj.__value_attrs__()
        assert isinstance(result, tuple), f"__value_attrs__ must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
        return result
    elif hasattr(obj, '__all_attrs__'):
        result = obj.__all_attrs__()
        assert isinstance(result, tuple), f"__all_attrs__ must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
        return result
    if dataclasses.is_dataclass(obj):
        if hasattr(obj, 'value_attrs'):
            result = obj.value_attrs
            assert isinstance(result, tuple), f"dataclass.value_attrs must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
            return result
        return all_attributes(obj)
    raise ValueError(f"{type(obj).__name__} must implement '__value_attrs__()' or be a dataclass to be used with value functions.")


def all_attributes(obj, assert_any=False) -> Tuple[str, ...]:
    if hasattr(obj, '__all_attrs__'):
        result = obj.__all_attrs__()
        assert isinstance(result, tuple), f"__value_attrs__ must return Tuple[str,...] but got '{type(result)}' from '{type(obj)}'"
        return result
    if not isinstance(obj, PhiTreeNode):
        raise ValueError(f"Not a PhiTreeNode: {type(obj).__name__}")
    result = set()
    if hasattr(obj, '__variable_attrs__'):
        result.update(obj.__variable_attrs__())
    if hasattr(obj, '__value_attrs__'):
        result.update(obj.__value_attrs__())
    if dataclasses.is_dataclass(obj) and not hasattr(obj, '__variable_attrs__') and not hasattr(obj, '__value_attrs__'):
        from ..dataclasses import data_fields
        result.update([f.name for f in data_fields(obj)])
    if assert_any:
        assert result, f"{type(obj).__name__} is not a valid tree node because it has no tensor-like attributes."
    return tuple(sorted(result))


def slice_(value: PhiTreeNodeType, slices: Union[Dict[str, Union[int, slice, str, tuple, list, Any]], Any]) -> PhiTreeNodeType:
    """
    Slices a `Tensor` or `phiml.math.magic.PhiTreeNode` along named dimensions.

    See Also:
        `unstack`.

    Args:
        value: `Tensor` or `phiml.math.magic.PhiTreeNode` or `Number` or `None`.
        slices: `dict` mapping dimension names to slices. A slice can be one of the following:

            * An index (`int`)
            * A range (`slice`)
            * An item name (`str`)
            * Multiple labels (comma-separated `str`)
            * Multiple indices or labels (`tuple` or `list`)

    Returns:
        `Tensor` or `phiml.math.magic.PhiTreeNode` of the same type as `value`.

    Examples:
        >>> math.slice([vec(x=0, y=1), vec(x=2, y=3)], {'vector': 'y'})
        [1, 3]
    """
    if slices is None:
        return value
    if isinstance(value, (bool, Number, str)) or value is None:
        return value
    if isinstance(value, tuple):
        return tuple([slice_(v, slices) for v in value])
    if isinstance(value, list):
        return [slice_(v, slices) for v in value]
    if isinstance(value, dict):
        return {k: slice_(v, slices) for k, v in value.items()}
    if isinstance(value, SHAPE_TYPES):
        return value.after_gather(slices)
    if value is range:
        from ._tensors import Tensor
        if isinstance(slices, Tensor):
            return slices
        raise NotImplementedError("range only supported for index slicing")
    if hasattr(value, '__getitem__'):
        return value[slices]
    if isinstance(value, PhiTreeNode):
        attrs = {key: getattr(value, key) for key in all_attributes(value)}
        new_attrs = {k: slice_(v, slices) for k, v in attrs.items()}
        return copy_with(value, **new_attrs)
    raise ValueError(f"value must be a PhiTreeNode but got {type(value)}")


MISSING_TENSOR = '__missing__'
NATIVE_TENSOR = '__native__'


@dataclasses.dataclass
class DataclassTreeNode:
    cls: type
    attr_type: Callable
    extracted: Dict[str, Any]  # trees without variable tensors
    not_extracted: Dict[str, Any]  # original values of non-extracted properties
    cache: Dict[str, Any]


def disassemble_dataclass(obj, attr_type=variable_attributes):
    extract_names = attr_type(obj)
    keys = {}
    values = []
    for attr in extract_names:
        key, value = disassemble_tree(getattr(obj, attr), False, attr_type)
        keys[attr] = key
        values.extend(value)
    non_attributes = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if f.name not in keys}
    from ..dataclasses._dep import get_unchanged_cache
    cache = get_unchanged_cache(obj, set(extract_names))
    return DataclassTreeNode(type(obj), attr_type, keys, non_attributes, cache), values


def assemble_dataclass(container: DataclassTreeNode, values: List[Tensor]):
    extracted = {a: assemble_tree(v, values, container.attr_type) for a, v in container.extracted.items()}
    instance = container.cls.__new__(container.cls)
    instance.__init__(**extracted, **container.not_extracted)
    if container.cache:
        instance.__dict__.update(container.cache)
    return instance


def disassemble_tree(obj: PhiTreeNodeType, cache: bool, attr_type=variable_attributes) -> Tuple[PhiTreeNodeType, List[Tensor]]:
    """
    Splits a nested structure of Tensors into the structure without the tensors and an ordered list of tensors.
    Native tensors will be wrapped in phiml.math.Tensors with default dimension names and dimension types `None`.

    See Also:
        `assemble_tree()`

    Args:
        obj: Nested structure of `Tensor` objects.
            Nested structures include: `tuple`, `list`, `dict`, `phiml.math.magic.PhiTreeNode`.
        cache: Whether to return cached versions of the tensors. This may reduce the number of native tensors required.

    Returns:
        empty structure: Same structure as `obj` but with the tensors replaced by `None`.
        tensors: Ordered `list` of all contained `Tensor` objects.
    """
    if obj is None:
        return MISSING_TENSOR, []
    elif isinstance(obj, Layout):
        keys, values = disassemble_tree(obj._obj, cache, attr_type)
        return {'__layout__': 1, 'stack_dim': obj._stack_dim._to_dict(False), 'obj': keys}, values
    elif isinstance(obj, Tensor):
        return None, [obj._cached() if cache else obj]
    elif isinstance(obj, Shape):
        return obj, []
    elif isinstance(obj, (tuple, list)):
        keys = []
        values = []
        for item in obj:
            key, value = disassemble_tree(item, cache, attr_type)
            keys.append(key)
            values.extend(value)
        return (tuple(keys) if isinstance(obj, tuple) else keys), values
    elif isinstance(obj, dict):
        keys = {}
        values = []
        for name, item in obj.items():
            key, value = disassemble_tree(item, cache, attr_type)
            keys[name] = key
            values.extend(value)
        return keys, values
    elif dataclasses.is_dataclass(obj):
        container, values = disassemble_dataclass(obj, attr_type=attr_type)
        if cache:
            from ._ops import cached
            values = [cached(v) for v in values]
        return container, values
    elif isinstance(obj, PhiTreeNode):
        attributes = attr_type(obj)
        keys = {}
        values = []
        for attr in attributes:
            key, value = disassemble_tree(getattr(obj, attr), cache, attr_type)
            keys[attr] = key
            values.extend(value)
        return copy_with(obj, **keys), values
    else:  # native tensor?
        try:
            backend = choose_backend(obj)
            if backend == OBJECTS:
                return obj, []
            sizes = backend.staticshape(obj)
            dims = [Dim(f"dim{i}", s, CHANNEL_DIM, None) for i, s in enumerate(sizes)]
            shape = PureShape(CHANNEL_DIM, {dim.name: dim for dim in dims})
            return NATIVE_TENSOR, [Dense(obj, shape.names, shape, backend)]
        except NoBackendFound:
            return obj, []


def assemble_tree(obj: PhiTreeNodeType, values: List[Tensor], attr_type=variable_attributes) -> PhiTreeNodeType:
    """ Reverses `disassemble_tree()` given an empty nested structure and a list of tensors. """
    if isinstance(obj, str) and obj == MISSING_TENSOR:
        return None
    elif isinstance(obj, str) and obj == NATIVE_TENSOR:
        value = values.pop(0)
        assert isinstance(value, Dense), f"Failed to assemble tree structure. Encountered {value}"
        if isinstance(value._native, np.ndarray) and value.shape == EMPTY_SHAPE:  # this can be represented as a Python scalar, which leads to less conversion errors
            return value._native.item()
        return value._native
    elif obj is None:
        value = values.pop(0)
        assert isinstance(value, Tensor)
        return value
    elif isinstance(obj, Shape):
        return obj
    elif isinstance(obj, list):
        return [assemble_tree(item, values, attr_type) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([assemble_tree(item, values, attr_type) for item in obj])
    elif isinstance(obj, dict) and '__layout__' in obj:
        content = assemble_tree(obj['obj'], values, attr_type)
        return Layout(content, Shape._from_dict(obj['stack_dim']))
    elif isinstance(obj, dict):
        return {name: assemble_tree(val, values, attr_type) for name, val in obj.items()}
    elif isinstance(obj, Tensor):
        return obj
    elif dataclasses.is_dataclass(obj):
        if isinstance(obj, DataclassTreeNode):
            return assemble_dataclass(obj, values)
    if isinstance(obj, PhiTreeNode):
        attributes = attr_type(obj)
        values = {a: assemble_tree(getattr(obj, a), values, attr_type) for a in attributes}
        return copy_with(obj, **values)
    return obj


def attr_paths(obj: PhiTreeNodeType, attr_type: Callable, root: str) -> List[str]:
    if obj is None:
        return []
    elif isinstance(obj, Layout):
        return attr_paths(obj._obj, attr_type, f'{root}._obj')
    elif isinstance(obj, Tensor):
        return [root]
    elif isinstance(obj, Shape):
        return []
    elif isinstance(obj, (tuple, list)):
        paths = []
        for i, item in enumerate(obj):
            path = attr_paths(item, attr_type, f'{root}[{i}]')
            paths.extend(path)
        return paths
    elif isinstance(obj, dict):
        paths = []
        for name, item in obj.items():
            path = attr_paths(item, attr_type, f'{root}[{name}]')
            paths.extend(path)
        return paths
    elif isinstance(obj, PhiTreeNode):
        attributes = attr_type(obj)
        paths = []
        for attr in attributes:
            path = attr_paths(getattr(obj, attr), attr_type, f'{root}.{attr}')
            paths.extend(path)
        return paths
    else:  # native tensor?
        try:
            return [] if choose_backend(obj) == OBJECTS else [root]
        except NoBackendFound:
            return []


def attr_paths_from_container(obj: PhiTreeNodeType, attr_type: Callable, root: str) -> List[str]:
    if isinstance(obj, str) and obj == MISSING_TENSOR:
        return []
    elif isinstance(obj, str) and obj == NATIVE_TENSOR:
        return [root]
    elif obj is None:
        return [root]
    elif isinstance(obj, Shape):
        return []
    elif isinstance(obj, (tuple, list)):
        return sum([attr_paths_from_container(v, attr_type, f'{root}[{i}]') for i, v in enumerate(obj)], [])
    elif isinstance(obj, dict) and '__layout__' in obj:
        return attr_paths_from_container(obj['obj'], attr_type, f'{root}._obj')
    elif isinstance(obj, dict):
        return sum([attr_paths_from_container(v, attr_type, f'{root}[{k}]') for k, v in obj.items()], [])
    elif isinstance(obj, Tensor):
        raise RuntimeError("Tensor found in container. This should have been set to None by disassemble_tree()")
    elif dataclasses.is_dataclass(obj):
        if isinstance(obj, DataclassTreeNode):
            assert attr_type == obj.attr_type
            return sum([attr_paths_from_container(v, attr_type, f'{root}.{k}') for k, v in obj.extracted.items()], [])
    if isinstance(obj, PhiTreeNode):
        attributes = attr_type(obj)
        return sum([attr_paths_from_container(getattr(obj, k), attr_type, f'{root}.{k}') for k in attributes], [])
    return []


def tree_map(f, tree, attr_type=value_attributes,
             include_non_attrs=True,
             treat_layout_as_leaf=False,
             treat_shapes_as_leaf=False,
             op_name: str = None,
             **f_kwargs):
    """
    Recursively iterates over Layouts, lists, tuples, dicts and the value attributes of PhiTreeNodes.
    Calls `f` on `Tensor` instances and everything else not in the above list.

    Args:
        f: Function mapping `Tensor`s or leaves
        tree: Nested tree or leaf
        attr_type: Which attributes to use for PhiTreeNodes. Typically, either `value_attributes` or `variable_attributes`.
        include_non_attrs: Whether to invoke `f` on non-attribute types, such as `str`, `Shape` or primitives.
        treat_layout_as_leaf: Whether to call `f` directly on `Layout` instances instead of their children.
        **f_kwargs: Keyword arguments for `f`.

    Returns:
        Tree matching structure of `tree` with transformed leaf values.
    """
    if isinstance(tree, Layout):
        if treat_layout_as_leaf:
            return f(tree, **f_kwargs)
        else:
            return tree._op1(lambda x: tree_map(f, x, attr_type, include_non_attrs, treat_layout_as_leaf, op_name, **f_kwargs), op_name or f.__name__)
    if isinstance(tree, Tensor) or tree is None:
        return f(tree, **f_kwargs)
    if isinstance(tree, list):
        return [tree_map(f, e, attr_type, include_non_attrs, treat_layout_as_leaf, **f_kwargs) for e in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(f, e, attr_type, include_non_attrs, treat_layout_as_leaf, **f_kwargs) for e in tree])
    elif isinstance(tree, dict):
        return {k: tree_map(f, e, attr_type, include_non_attrs, treat_layout_as_leaf, **f_kwargs) for k, e in tree.items()}
    elif isinstance(tree, Shape):
        return f(tree, **f_kwargs) if treat_shapes_as_leaf else tree
    elif isinstance(tree, PhiTreeNode):
        attrs = {key: getattr(tree, key) for key in attr_type(tree)}
        new_attrs = {k: tree_map(f, v, attr_type, include_non_attrs, treat_layout_as_leaf, **f_kwargs) for k, v in attrs.items()}
        return copy_with(tree, **new_attrs)
    else:
        from ..dataclasses._dataclasses import NON_ATTR_TYPES
        if include_non_attrs or not isinstance(tree, NON_ATTR_TYPES):
            return f(tree, **f_kwargs)  # try anyway
        return tree


def tree_map_with_paths(f, tree, root: str, attr_type=all_attributes, treat_layout_as_leaf=False, tensor_classes=(Tensor,)):
    paths, leaves = [], []
    list_leaves(tree, root, paths, leaves, attr_type, treat_layout_as_leaf=treat_layout_as_leaf, tensor_classes=tensor_classes)
    outputs = []
    for leaf, path in zip(leaves, paths):
        outputs.append(f(path, leaf))
    return replace_all_leaves(tree, outputs, attr_type, treat_layout_as_leaf=treat_layout_as_leaf, tensor_classes=tensor_classes)


def list_leaves(obj, root: str, paths: list, leaves: list, attr_type=all_attributes, treat_layout_as_leaf=False, tensor_classes=(Tensor,)):
    if isinstance(obj, Layout):
        if treat_layout_as_leaf:
            if isinstance(obj, tensor_classes):
                paths.append(root)
                leaves.append(obj)
        else:  # recurse into layout
            list_leaves(obj._obj, f'{root}._obj', paths, leaves, attr_type, treat_layout_as_leaf, tensor_classes)
    elif isinstance(obj, tensor_classes):
        paths.append(root)
        leaves.append(obj)
    elif isinstance(obj, (tuple, list)):
        for i, item in enumerate(obj):
            list_leaves(item, f'{root}[{i}]', paths, leaves, attr_type, treat_layout_as_leaf, tensor_classes)
    elif isinstance(obj, dict):
        for name, item in obj.items():
            list_leaves(item, f'{root}[{name}]', paths, leaves, attr_type, treat_layout_as_leaf, tensor_classes)
    elif dataclasses.is_dataclass(obj) or isinstance(obj, PhiTreeNode):
        attributes = attr_type(obj)
        for attr in attributes:
            list_leaves(getattr(obj, attr), f'{root}.{attr}', paths, leaves, attr_type, treat_layout_as_leaf, tensor_classes)


def replace_all_leaves(obj, leaves: list, attr_type=all_attributes, treat_layout_as_leaf=False, tensor_classes=(Tensor,)):
    if obj is None:
        return None
    elif isinstance(obj, Layout):
        if treat_layout_as_leaf:
            if isinstance(obj, tensor_classes):
                return leaves.pop(0)
        else:  # recurse into layout
            return Layout(replace_all_leaves(obj._obj, leaves, attr_type, treat_layout_as_leaf, tensor_classes), obj._stack_dim)
    elif isinstance(obj, tensor_classes):
        return leaves.pop(0)
    elif isinstance(obj, tuple):
        return tuple([replace_all_leaves(item, leaves, attr_type, treat_layout_as_leaf, tensor_classes) for item in obj])
    elif isinstance(obj, (tuple, list)):
        return [replace_all_leaves(item, leaves, attr_type, treat_layout_as_leaf, tensor_classes) for item in obj]
    elif isinstance(obj, dict):
        return {k: replace_all_leaves(v, leaves, attr_type, treat_layout_as_leaf, tensor_classes) for k, v in obj.items()}
    elif dataclasses.is_dataclass(obj) or isinstance(obj, PhiTreeNode):
        attributes = attr_type(obj)
        new_vals = {attr: replace_all_leaves(getattr(obj, attr), leaves, attr_type, treat_layout_as_leaf, tensor_classes) for attr in attributes}
        return replace(obj, **new_vals)
    return obj


def get_value_by_path(path: str, obj):
    """path needs to start with either '.' or '[' or be empty"""
    if not path:
        return obj
    name = path[1:].split('.')[0].split('[')[0]
    if path.startswith('.'):
        return get_value_by_path(path[1+len(name):], getattr(obj, name))
    elif path.startswith('['):
        name = name[:-1]  # remove trailing ']'
        idx = int(name) if isinstance(obj, (tuple, list)) else name
        return get_value_by_path(path[2+len(name):], obj[idx])
    raise ValueError(f"Illegal path: {path}")


def tree_broadcast(attr_type=value_attributes, include_non_attrs=True, treat_layout_as_leaf=False, treat_shapes_as_leaf=False):
    def wrapper(f):
        def partial_tree_map(tree, **f_kwargs):
            return tree_map(f, tree, attr_type=attr_type, include_non_attrs=include_non_attrs, treat_layout_as_leaf=treat_layout_as_leaf, treat_shapes_as_leaf=treat_shapes_as_leaf, **f_kwargs)
        return partial_tree_map
    return wrapper


def save(file: Union[Tensor, str], obj: PhiTreeNodeType, mkdir=True):
    """
    Saves a `Tensor` or tree using NumPy.
    This function converts all tensors contained in `obj` to NumPy tensors before storing.
    Each tensor is given a name corresponding to its path within `obj`, allowing reading only specific arrays from the file later on.
    Pickle is used for structures, but no reference to `Tensor` or its sub-classes is included.

    Examples:

        >>> B = batch(b=3)
        >>> files = -f-f"data/test_{arange(B)}.npz"
        >>> data = randn(B, spatial(x=10))
        >>> save(files, data)  # store 10 values per file
        >>> assert_close(data, load(files))

    See Also:
        `load()`.

    Args:
        file: Either single file to read as `str` or a batch of files as a string `Tensor`. The file ending will be completed to `.npz`.
            When a batch of paths is provided, the data `obj` is sliced along the dims of `file` and broken up to be stored among the multiple files.
            For obtaining a batch of files, see `wrap()`, `phiml.os.listdir()`, `phiml.math.f`.
        obj: `Tensor` or tree to store.
        mkdir: Whether to create the file's directory if it doesn't exist.
    """
    tree, tensors = disassemble_tree(obj, False, all_attributes)
    paths = attr_paths(obj, all_attributes, 'root')
    assert len(paths) == len(tensors)
    for idx in shape(file).meshgrid():
        file_i = file[idx].native() if isinstance(file, Tensor) else file
        tensors_i = [t[idx] for t in tensors] if idx else tensors
        natives = [t._natives() for t in tensors_i]
        specs = [serialize_spec(t._spec_dict()) for t in tensors_i]
        native_paths = [[f'{p}:{i}' for i in range(len(ns))] for p, ns in zip(paths, natives)]
        all_natives = sum(natives, ())
        all_paths = sum(native_paths, [])
        all_np = [choose_backend(n).numpy(n) for n in all_natives]
        if mkdir and os.path.dirname(file_i):
            os.makedirs(os.path.dirname(file_i), exist_ok=True)
        np.savez(file_i, tree=np.asarray({'tree': tree}, dtype=object), specs=specs, paths=paths, **{p: n for p, n in zip(all_paths, all_np)})


def load(file: Union[str, Tensor]):
    """
    Loads a `Tensor` or tree from one or multiple files previously written using `save`.

    All tensors are restored as NumPy arrays, not the backend-specific tensors they may have been written as.
    Use `convert()` to convert all or some of the tensors to a different backend.

    Examples:

        >>> B = batch(b=3)
        >>> files = -f-f"data/test_{arange(B)}.npz"
        >>> data = randn(B, spatial(x=10))
        >>> save(files, data)  # store 10 values per file
        >>> assert_close(data, load(files))

    See Also:
        `save()`.

    Args:
        file: Either single file to read as `str` or a batch of files as a string `Tensor`.
            When a batch of paths is provided, each file is loaded and the results are stacked according to the dims of `file`.
            For obtaining a batch of files, see `wrap()`, `phiml.os.listdir()`, `phiml.math.f`.

    Returns:
        Same type as what was written.
    """
    def load_single(file: str):
        data = np.load(file, allow_pickle=True)
        all_np = {k: data[k] for k in data if k not in ['tree', 'specs', 'paths']}
        specs = [unserialize_spec(spec) for spec in data['specs'].tolist()]
        tensors = assemble_tensors(list(all_np.values()), specs)
        tree = data['tree'].tolist()['tree']  # this may require outside classes via pickle
        stored_paths = data['paths'].tolist()
        new_paths = attr_paths_from_container(tree, all_attributes, 'root')
        if tuple(stored_paths) != tuple(new_paths):
            lookup = {path: t for path, t in zip(stored_paths, tensors)}
            tensors = [lookup[p] for p in new_paths]
        return assemble_tree(tree, tensors, attr_type=all_attributes)
    if isinstance(file, str):
        return load_single(file)
    from ._functional import map_
    return map_(load_single, file)


def save_h5(file: Union[Tensor, str], obj: PhiTreeNodeType, mkdir=True):
    """
    Save structured data as HDF5.

    See Also:
        `load_h5()`.

    Args:
        file: Either single file (`str`) or a batch of files (`Tensor[str]`). The file ending will be completed to `.npz`.
            When a batch of paths is provided, the data of multiple files is stacked.
            For obtaining a batch of files, see `wrap()`, `phiml.os.listdir()`, `phiml.math.f`.
        obj: `Tensor` or tree to store.
        mkdir: Whether to create the file's directory if it doesn't exist.
    """
    from h5py import File
    tree_str, paths, dense_tensors = serialize_tree(obj, 'root')
    for idx in shape(file).meshgrid():
        file_i = file[idx].native() if isinstance(file, Tensor) else file
        tensors_i = [t[idx] for t in dense_tensors] if idx else dense_tensors
        if mkdir and os.path.dirname(file_i):
            os.makedirs(os.path.dirname(file_i), exist_ok=True)
        with File(file_i, mode='w') as f:
            f.attrs['content_hierarchy'] = tree_str
            for path, t in zip(paths, tensors_i):
                f.create_dataset(path, data=t.numpy(t._names))
                f.attrs[path + '_shape'] = to_spec_str(t.shape)
                f.attrs[path + '_names'] = ",".join(t._names)
                f.attrs[path + '_dtype'] = str(t.dtype)


def load_h5(file: Union[Tensor, str], structure=None, load_into_memory=('root',)):
    """
    Read structured data from an HDF5 file.

    See Also:
        `save_h5()`.

    Args:
        file: Either single HDF5 file (`str`) or a batch of files (`Tensor[str]`).
            When a batch of paths is provided, the data of multiple files is stacked.
            For obtaining a batch of files, see `wrap()`, `phiml.os.listdir()`, `phiml.math.f`.
        structure: dataclass of pytree. The structure into which the values should be loaded. If `None`, loads objects as `dict`.
        load_into_memory: Paths specifying which tensors to load. Others will be returned as references to the location within the H5 file.
            All paths start with `'root'`. For example, to load only the first list entry of the `'location'` dict entry, set `load_into_memory=('root[location][0]',)`

    Returns:
        Data contained in the H5 file. If `structure` is specified, the result will match that type and layout.
    """
    assert structure is None, f"structure not yet supported"
    from ..parallel._tensor_cache import H5Source, DiskTensor
    def load_single(file: str):
        source = H5Source(file)
        f = source.h5file
        tree = f.attrs['content_hierarchy']
        paths: list[str] = [a[:-6] for a in f.attrs if a.endswith('_shape')]
        tensors = {path: DiskTensor(source, path, {}, [s for s in f.attrs[path+'_names'].split(',') if s], from_spec_str(f.attrs[path+'_shape']), NUMPY, DType.from_name(f.attrs[path+'_dtype'])) for path in paths}
        tensors = {path: t.as_persistent() if path.startswith(load_into_memory) else t for path, t in tensors.items()}
        return unserialize_tree(tree, 'root', tensors)
    if isinstance(file, str):
        return load_single(file)
    from ._functional import map_
    return map_(load_single, file)


def serialize_tree(obj, root: str):
    if isinstance(obj, Dense):
        return "D", [root], [obj]
    elif isinstance(obj, TensorStack):
        trees, paths, tensors = zip(*[serialize_tree(t, f"{root}[{i}]") for i, t in enumerate(obj._tensors)])
        return f"stack({to_spec_str(obj._stack_dim)},{','.join(trees)})", sum(paths, []), sum(tensors, [])
    elif isinstance(obj, Tensor):
        raise NotImplementedError(obj)
    elif isinstance(obj, dict):
        trees, paths, tensors = zip(*[serialize_tree(t, f"{root}[{k}]") for k, t in obj.items()])
        return "{" + ",".join([f"{k}={t}" for k, t in zip(obj, trees)]) + "}", sum(paths, []), sum(tensors, [])
    raise NotImplementedError(obj)


def unserialize_tree(tree: str, root: str, data_by_path: dict[str, Tensor]):
    if tree == 'D':
        return data_by_path[root]
    elif tree.startswith('stack'):
        stack_dim, *args = split_top_level_comma(tree[6:-1])
        components = [unserialize_tree(arg, f"{root}[{i}]", data_by_path) for i, arg in enumerate(args)]
        return TensorStack(components, from_spec_str(stack_dim))
    elif tree.startswith('{') and tree.endswith('}'):
        entries = split_top_level_comma(tree[1:-1])
        result = {}
        for entry in entries:
            name, val = entry.split('=', 1)
            result[name] = unserialize_tree(val, f"{root}[{name}]", data_by_path)
        return result
    raise NotImplementedError(tree)


def find_differences(tree1, tree2, compare_tensors_by_id=False, attr_type=value_attributes, tensor_equality=None) -> Sequence[Tuple[str, str, Any, Any]]:
    """
    Compares `tree1` and `tree2` and returns all differences in the form `(difference_description: str, variable_identifier: str, value1, value2)`.

    Args:
        tree1: Nested tree or leaf
        tree2: Nested tree or leaf
        compare_tensors_by_id: Whether `phiml.math.Tensor` objects should be compared by identity or values.
        attr_type: What attributes to compare, either `value_attributes` or `variable_attributes`.
        tensor_equality: Function that compares two tensors for equality. `None` defaults to `equal`.

    Returns:
        List of differences, each represented as a `tuple`.
    """
    result = []
    _recursive_diff(tree1, tree2, '', result, compare_tensors_by_id, attr_type, tensor_equality)
    return result


def _recursive_diff(a, b, path: str, result: list, compare_tensors_by_id=False, attr_type=value_attributes, tensor_equality=None):
    if a is b:
        return
    if (a is None) != (b is None):
        result.append(("Only one tree has a value, other is None", path, a, b))
        return
    if isinstance(a, Layout):
        if not isinstance(b, Layout):
            result.append((f"Only one value is a Layout, the other is {type(b)}", path, a, b))
        else:
            _recursive_diff(a._obj, b._obj, path, result, compare_tensors_by_id, attr_type, tensor_equality)
    elif isinstance(b, Layout):
        result.append((f"Only one value is a Layout, the other is {type(a)}", path, a, b))
    elif isinstance(a, Tensor) or isinstance(b, Tensor):
        if not isinstance(a, Tensor) or not isinstance(b, Tensor):
            result.append((f"Only one value is a Tensor: {type(a).__name__} vs {type(b).__name__}", path, a, b))
            return
        if compare_tensors_by_id:
            if a is not b:
                result.append(("Tensor ids do not match", path, a, b))
        else:
            if tensor_equality is None:
                from ._ops import equal
                tensor_equality = partial(equal, equal_nan=True)
            if a.shape != b.shape:
                result.append((f"Tensor shapes do not match: {a.shape} vs {b.shape}", path, a, b))
            elif not tensor_equality(a, b):
                result.append((f"Tensor values do not match", path, a, b))
    elif type(a) != type(b):
        result.append((f"Types do not match: {type(a).__name__} vs {type(b).__name__}", path, a, b))
        return
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            result.append((f"Lengths do not match: {len(a)} vs {len(b)}", path, a, b))
        else:
            for i, (ae, be) in enumerate(zip(a, b)):
                _recursive_diff(ae, be, f"{path}[{i}]", result, compare_tensors_by_id, attr_type, tensor_equality)
    elif isinstance(a, dict):
        if set(a) != set(b):
            result.append((f"Keys do not match: {set(a)} vs {set(b)}", path, a, b))
        else:
            for k, av in a.items():
                bv = b[k]
                _recursive_diff(av, bv, f"{path}[{k}]", result, compare_tensors_by_id, attr_type, tensor_equality)
    elif isinstance(a, PhiTreeNode):
        a_attrs = attr_type(a)
        if set(a_attrs) != set(attr_type(b)):
            result.append((f"Available properties do not match: {set(a_attrs)} vs {set(attr_type(b))}", path, a, b))
        else:
            for k in a_attrs:
                av = getattr(a, k)
                bv = getattr(b, k)
                _recursive_diff(av, bv, f"{path}.{k}", result, compare_tensors_by_id, attr_type, tensor_equality)
    else:
        try:
            backend = choose_backend(a, b)
            if backend.shape(a) != backend.shape(b):
                result.append((f"Shapes do not match: {backend.shape(a)} vs {backend.shape(b)}", path, a, b))
            else:
                equal_tensor = backend.equal(a, b) | (backend.isnan(a) & backend.isnan(b))
                equal = backend.numpy(backend.all(equal_tensor))
                if not equal:
                    result.append(("Values do not match", path, a, b))
        except NoBackendFound:
            if a != b:
                result.append(("Values do not match", path, a, b))
