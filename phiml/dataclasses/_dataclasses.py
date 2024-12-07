import collections
import dataclasses
import inspect
from functools import cached_property
from typing import TypeVar, Callable, Tuple, List, Set, Iterable, Optional, get_origin, get_args, Dict, Sequence

from phiml.dataclasses._dep import get_unchanged_cache
from phiml.math import DimFilter, shape, Shape
from phiml.math._magic_ops import slice_
from phiml.math.magic import slicing_dict, BoundDim

PhiMLDataclass = TypeVar("PhiMLDataclass")


def dataclass(cls=None, /, *, getitem=True, dim_attrs=True, keepdims=None, dim_repr=True,
              init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=True, match_args=True, kw_only=False, slots=False, weakref_slot=False):
    """
    Convenience decorator for PhiML dataclasses.
    This builds a regular dataclass but adds additional options for slicing.

    If you don't require the additional features or want to implement them yourself, you may also use `@dataclass` from the `dataclasses` module.

    Args:
        getitem: Whether to generate the `__getitem__` method for slice / gather / boolean_mask, depending on the argument.
        dim_attrs: Whether to generate `__getattr__` that allows slicing via the syntax `instance.dim[...]` where `dim` is the name of any dim present on `instance`.
        keepdims: Which dimensions should be kept with size 1 taking a single slice along them. This will preserve item names.
        dim_repr: Whether to replace the default `repr` of a dataclass by a simplified one based on the object's shape.

    All other args are passed on to `dataclasses.dataclass`.
    Note however, that `frozen` must be `True` which is the default in this decorator.
    """

    assert frozen, f"PhiML dataclasses must be frozen."

    def wrap(cls):
        overridden = [pair[0] for pair in inspect.getmembers(cls, inspect.isfunction)]
        dataclasses.dataclass(cls, init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash, frozen=frozen, match_args=match_args, kw_only=kw_only, slots=slots, weakref_slot=weakref_slot)
        assert attributes(cls), f"PhiML dataclasses must have at least one field storing a Shaped object, such as a Tensor, tree of Tensors or compatible dataclass."
        if getitem and '__getitem__' not in overridden:
            def __dataclass_getitem__(obj, item):
                return getitem_impl(obj, item, keepdims=keepdims)
            cls.__getitem__ = __dataclass_getitem__
        if dim_attrs and '__getattr__' not in overridden:
            def __dataclass_getattr__(obj, name: str):
                if name in ('shape', '__shape__', '__all_attrs__', '__variable_attrs__', '__value_attrs__'):  # these can cause infinite recursion
                    raise AttributeError(f"'{type(obj)}' instance has no attribute '{name}'")
                if name in shape(obj):
                    return BoundDim(obj, name)
                else:
                    raise AttributeError(f"'{type(obj)}' instance has no attribute '{name}'")
            cls.__getattr__ = __dataclass_getattr__
        if dim_repr and '__repr__' not in overridden:
            def __dataclass_repr__(obj):
                try:
                    obj_shape = shape(obj)
                except BaseException as err:
                    obj_shape = f"Unknown shape: {type(err).__name__}"
                return f"{type(obj).__name__}[{obj_shape}]"
            cls.__repr__ = __dataclass_repr__
        return cls

    if cls is None:  # See if we're being called as @dataclass or @dataclass().
        return wrap
    return wrap(cls)


NON_ATTR_TYPES = str, int, float, complex, bool, Shape, slice, Callable


def attributes(obj) -> Sequence[dataclasses.Field]:
    """
    List all dataclass Fields of `obj` that are considered an attribute, i.e. could possibly hold (directly or indirectly) a `Tensor`.

    Args:
        obj: Dataclass type or instance.

    Returns:
        Sequence of `dataclasses.Field`.
    """
    return [f for f in dataclasses.fields(obj) if _is_child_field(f)]


def _is_child_field(field: dataclasses.Field):
    primitives = _get_primitive_types(field.type)
    return any(p not in NON_ATTR_TYPES for p in primitives)


def _get_primitive_types(field_type) -> list:
    """Returns None for unknown types."""
    if field_type is Ellipsis:
        return []
    origin_type = get_origin(field_type)
    if origin_type in {list, List, tuple, Tuple, set, Set, Iterable, Optional, collections.abc.Sequence}:
        args = get_args(field_type)  # The arguments passed to the generic (e.g., List[int] -> (int,))
        return sum([_get_primitive_types(a) for a in args], []) if args else [None]
    elif origin_type in {Dict, dict}:
        k_type, v_type = get_args(field_type)
        return _get_primitive_types(v_type)
    else:
        return [field_type]


def replace(obj: PhiMLDataclass, /, call_metaclass=False, **changes) -> PhiMLDataclass:
    """
    Create a copy of `obj` with some fields replaced.
    Unlike `dataclasses.replace()`, this function also transfers `@cached_property` members if their dependencies are not affected.

    Args:
        obj: Dataclass instance.
        call_metaclass: Whether to copy `obj` by invoking `type(obj).__call__`.
            If `obj` defines a metaclass, this will allow users to define custom constructors for dataclasses.
        **changes: New field values to replace old ones.

    Returns:
        Copy of `obj` with replaced values.
    """
    cls = obj.__class__
    kwargs = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    kwargs.update(**changes)
    if call_metaclass:
        new_obj = cls(**kwargs)
    else:  # This allows us override the dataclass constructor with a metaclass for user convenience, but not call it internally.
        new_obj = cls.__new__(cls)
        new_obj.__init__(**kwargs)
    cache = get_unchanged_cache(obj, set(changes.keys()))
    new_obj.__dict__.update(cache)
    return new_obj


def getitem(obj: PhiMLDataclass, item, keepdims: DimFilter = None) -> PhiMLDataclass:
    """
    Slice / gather a dataclass by broadcasting the operation to its attributes.

    You may call this from `__getitem__` to allow the syntax `my_class[component_str]`, `my_class[slicing_dict]`, `my_class[boolean_tensor]` and `my_class[index_tensor]`.

    ```python
    def __getitem__(self, item):
        return getitem(self, item)
    ```

    Args:
        obj: Dataclass instance to slice / gather.
        item: One of the supported tensor slicing / gathering values.
        keepdims: Dimensions that will not be removed during slicing.
            When selecting a single slice, these dims will remain with size 1.

    Returns:
        Slice of `obj` of same type.
    """
    assert dataclasses.is_dataclass(obj), f"obj must be a dataclass but got {type(obj)}"
    item = slicing_dict(obj, item)
    if keepdims:
        keep = shape(obj).only(keepdims)
        for dim, sel in item.items():
            if dim in keep:
                if isinstance(sel, int):
                    item[dim] = slice(sel, sel+1)
                elif isinstance(sel, str) and ',' not in sel:
                    item[dim] = [sel]
    if not item:
        return obj
    attrs = attributes(obj)
    kwargs = {f.name: slice_(getattr(obj, f.name), item) if f in attrs else getattr(obj, f.name) for f in dataclasses.fields(obj)}
    cls = type(obj)
    new_obj = cls.__new__(cls, **kwargs)
    new_obj.__init__(**kwargs)
    cache = {k: slice_(v, item) for k, v in obj.__dict__.items() if isinstance(getattr(type(obj), k, None), cached_property) and not isinstance(v, Shape)}
    new_obj.__dict__.update(cache)
    return new_obj


getitem_impl = getitem


def stack(objs: Sequence, dim):
    raise NotImplementedError  # stack cached properties


def concat(obj: Sequence, dim):
    raise NotImplementedError  # concat cached properties


def expand(obj, dims):
    raise NotImplementedError  # expand cached Tensor properties
