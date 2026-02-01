import dataclasses
import warnings
from functools import partial
from numbers import Number
from typing import TypeVar, Tuple, Dict, Union, Optional, Sequence, Callable

import numpy as np

from . import channel
from ._shape import Shape, DimFilter, batch, instance, shape, non_batch, merge_shapes, spatial, dual, auto, parse_shape_spec, DIM_FUNCTIONS, \
    INV_CHAR, concat_shapes_, DEBUG_CHECKS, SHAPE_TYPES, primal, NotCompatible, replace_dims, _shape_replace
from ._tree import PhiTreeNodeType, all_attributes, copy_with, tree_map, value_attributes
from .magic import Sliceable, Shaped, Shapable, PhiTreeNode
from ..backend import choose_backend, NoBackendFound
from ..backend._dtype import DType

MagicType = TypeVar('MagicType')
OtherMagicType = TypeVar('OtherMagicType')


class MagicNotImplemented(Exception): pass


def unstack(value: MagicType, dim: DimFilter, expand=False) -> Tuple[MagicType, ...]:
    """
    Un-stacks a `Sliceable` along one or multiple dimensions.

    If multiple dims are given, the order of elements will be according to the dimension order in `dim`, i.e. elements along the last dimension will be neighbors in the returned `tuple`.
    If no dimension is given or none of the given dims exists on `value`, returns a list containing only `value`.

    See Also:
        `phiml.math.slice`.

    Args:
        value: `phiml.math.magic.Shapable`, such as `phiml.math.Tensor`
        dim: Dimensions as `Shape` or comma-separated `str` or dimension type, i.e. `channel`, `spatial`, `instance`, `batch`.
        expand: If `True`, `dim` must be a `Shape` and the returned tuple will have length `dim.volume`. Otherwise, only existing dims are unstacked.

    Returns:
        `tuple` of objects matching the type of `value`.

    Examples:
        >>> unstack(expand(0, spatial(x=5)), 'x')
        (0.0, 0.0, 0.0, 0.0, 0.0)
    """
    if DEBUG_CHECKS:
        assert isinstance(value, Sliceable) and isinstance(value, Shaped), f"Cannot unstack {type(value).__name__}. Must be Sliceable and Shaped, see https://tum-pbs.github.io/PhiML/phiml/math/magic.html"
    dims = shape(value).only(dim, reorder=True)
    if expand:
        assert isinstance(dim, Shape)
        if dim not in dims:
            value = expand_(value, dim)
        dims = dim
    if dims.rank == 0:
        return value,
    if dims.rank == 1:
        if hasattr(value, '__unstack__'):
            result = value.__unstack__(dims.names)
            if result is not NotImplemented:
                if DEBUG_CHECKS:
                    assert isinstance(result, tuple), f"__unstack__ must return a tuple but got {type(result)}"
                    assert all([isinstance(item, Sliceable) for item in result]), f"__unstack__ must return a tuple of Sliceable objects but not all items were sliceable in {result}"
                return result
        from ._tree import slice_
        return tuple([slice_(value, {dims.name: i}) for i in range(dims.size)])
    else:  # multiple dimensions
        if hasattr(value, '__pack_dims__'):
            packed_dim = batch('_unstack')
            value_packed = value.__pack_dims__(dims, packed_dim, pos=None)
            if value_packed is not NotImplemented:
                return unstack(value_packed, packed_dim)
        unstack_dim = _any_uniform_dim(dims)
        first_unstacked = unstack(value, unstack_dim)
        inner_unstacked = [unstack(v, dims.without(unstack_dim)) for v in first_unstacked]
        return sum(inner_unstacked, ())


def _any_uniform_dim(dims: Shape):
    for dim in dims:
        if dim.is_uniform:
            return dim
    raise ValueError(f"Uniform dimension required but found only non-uniform dims {dims}")


def stack(values: Union[Sequence[PhiTreeNodeType], Dict[str, PhiTreeNodeType]], dim: Union[Shape, str], expand_values=False, simplify=False, layout_non_matching=False, **kwargs) -> PhiTreeNodeType:
    """
    Stacks `values` along the new dimension `dim`.
    All values must have the same spatial, instance and channel dimensions. If the dimension sizes vary, the resulting tensor will be non-uniform.
    Batch dims will be added as needed.

    Stacking tensors is performed lazily, i.e. the memory is allocated only when needed.
    This makes repeated stacking and slicing along the same dimension very efficient, i.e. jit-compiled functions will not perform these operations.

    Args:
        values: Collection of `phiml.math.magic.Shapable`, such as `phiml.math.Tensor`
            If a `dict`, keys must be of type `str` and are used as labels along `dim`.
        dim: `Shape` with a least one dimension. None of these dims can be present with any of the `values`.
            If `dim` is a single-dimension shape, its size is determined from `len(values)` and can be left undefined (`None`).
            If `dim` is a multi-dimension shape, its volume must be equal to `len(values)`.
        expand_values: If `True`, will first add missing dims to all values, not just batch dimensions.
            This allows tensors with different dims to be stacked.
            The resulting tensor will have all dims that are present in `values`.
            If `False`, this may return a non-numeric object instead.
        simplify: If `True` and all values are equal, returns one value without adding the dimension.
        layout_non_matching: If non-matching values should be stacked using a Layout object, i.e. should be put into a named list instead.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dims to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dims must always work without keyword arguments.

    Returns:
        `Tensor` containing `values` stacked along `dim`.

    Examples:
        >>> stack({'x': 0, 'y': 1}, channel('vector'))
        (x=0, y=1)

        >>> stack([math.zeros(batch(b=2)), math.ones(batch(b=2))], channel(c='x,y'))
        (x=0.000, y=1.000); (x=0.000, y=1.000) (bᵇ=2, cᶜ=x,y)

        >>> stack([vec(x=1, y=0), vec(x=2, y=3.)], batch('b'))
        (x=1.000, y=0.000); (x=2.000, y=3.000) (bᵇ=2, vectorᶜ=x,y)
    """
    assert len(values) > 0, f"stack() got empty sequence {values}"
    if simplify and len(values) == 1:
        return next(iter(values.values())) if isinstance(values, dict) else values[0]
    if not dim:
        assert len(values) == 1, f"Only one element can be passed as `values` if no dim is passed but got {values}"
        return next(iter(values.values())) if isinstance(values, dict) else values[0]
    if not isinstance(dim, SHAPE_TYPES):
        dim = auto(dim)
    values_ = tuple(values.values()) if isinstance(values, dict) else values
    if simplify:
        if all(v is None for v in values_):
            return None
        if all(type(v) == type(values_[0]) for v in values_[1:]):
            from ._tensors import equality_by_shape_and_value
            with equality_by_shape_and_value(equal_nan=True):
                if all(v == values_[0] for v in values_[1:]):
                    return values_[0]
    shapes = [shape(v) for v in values_]
    if not expand_values:
        v0_dims = set(shapes[0].non_batch.names)
        for s in shapes[1:]:
            if set(s.non_batch.names) != v0_dims:  # shapes don't match
                if layout_non_matching:
                    from ._tensors import layout
                    return layout(values, dim)
                raise ValueError(f"Non-batch dims must match but got: {v0_dims} and {s.non_batch.names}. Manually expand tensors or set expand_values=True")
    # --- Add missing dims ---
    if expand_values:
        all_dims = merge_shapes(*shapes, allow_varying_sizes=True)
        if isinstance(values, dict):
            values = {k: expand(v, all_dims - s) for (k, v), s in zip(values.items(), shapes)}
        else:
            values = [expand(v, all_dims - s) for v, s in zip(values, shapes)]
    else:
        all_batch_dims = merge_shapes(*[s.batch for s in shapes], allow_varying_sizes=True)
        if isinstance(values, dict):
            values = {k: expand(v, all_batch_dims - s) for (k, v), s in zip(values.items(), shapes)}
        else:
            values = [expand(v, all_batch_dims - s) for v, s in zip(values, shapes)]
    if dim.rank == 1:
        assert dim.size == len(values) or dim.size is None, f"stack dim size must match len(values) or be undefined but got {dim} for {len(values)} values"
        if dim.size is None:
            dim = dim.with_size(len(values))
        if isinstance(values, dict):
            dim_labels = tuple([k.name if isinstance(k, SHAPE_TYPES) else k for k in values.keys()])
            assert all(isinstance(k, str) for k in dim_labels), f"dict keys must be of type str but got {dim_labels}"
            values = tuple(values.values())
            dim = dim.with_size(dim_labels)
        # --- First try __stack__ ---
        for v in values:
            if hasattr(v, '__stack__'):
                result = v.__stack__(values, dim, **kwargs)
                if result is not NotImplemented:
                    if DEBUG_CHECKS:
                        assert isinstance(result, SHAPE_TYPES) if isinstance(v, SHAPE_TYPES) else isinstance(result, Shapable), "__stack__ must return a Shapable object"
                    return result
        # --- Next: try stacking attributes for tree nodes ---
        if any(dataclasses.is_dataclass(v) for v in values):
            from ..dataclasses._merge import dc_stack
            try:
                return dc_stack(values, dim, expand_values=expand_values, simplify=simplify, layout_non_matching=layout_non_matching, **kwargs)
            except NotCompatible as err:
                if layout_non_matching:
                    from ._tensors import layout
                    return layout(values, dim)
                raise err
        if all(isinstance(v, dict) for v in values):
            keys = set(values[0])
            if all(set(v) == keys for v in values[1:]):
                new_dict = {}
                for k in keys:
                    k_values = [v[k] for v in values]
                    new_dict[k] = stack(k_values, dim, expand_values=expand_values, simplify=simplify, **kwargs)
                return new_dict
            raise NotImplementedError
        if any(isinstance(v, (tuple, list, dict)) for v in values_):
            from ._tensors import wrap, layout
            if _is_data_array(values_):
                tensors = [wrap(v) for v in values_]
                return stack(tensors, dim)
            elif all(isinstance(v, (tuple, list, dict)) for v in values_) and _contains_tensor(values_):
                if all(isinstance(v, (tuple, list)) for v in values_):
                    return [stack([v[i] for v in values_], dim, expand_values=expand_values, simplify=simplify, layout_non_matching=layout_non_matching, **kwargs) for i in range(len(values_[0]))]
                # the case of dicts is handled above
            else:
                assert len(dim) == 1, f"Cannot stack values with nested tuples, lists or dicts along multiple dimensions {dim}"
                return layout(values_, dim)
        if all(isinstance(v, PhiTreeNode) for v in values):
            attributes = all_attributes(values[0])
            if attributes and all(all_attributes(v) == attributes for v in values):
                new_attrs = {}
                for a in attributes:
                    a_values = [getattr(v, a) for v in values]
                    if all(v is a_values[0] for v in a_values[1:]):
                        new_attrs[a] = expand(a_values[0], dim, **kwargs) if a_values[0] is not None else a_values[0]
                    else:
                        new_attrs[a] = stack(a_values, dim, expand_values=expand_values, simplify=simplify, **kwargs)
                return copy_with(values[0], **new_attrs)
            else:
                warnings.warn(f"Failed to concat values using value attributes because attributes differ among values {values}")
        # --- Fallback: use expand and concat ---
        for v in values:
            if not hasattr(v, '__stack__') and hasattr(v, '__concat__') and hasattr(v, '__expand__'):
                expanded_values = tuple([expand(v, dim.with_size(1 if dim.labels[0] is None else dim.labels[0][i]), **kwargs) for i, v in enumerate(values)])
                if len(expanded_values) > 8:
                    warnings.warn(f"stack() default implementation is slow on large dims ({dim.name}={len(expanded_values)}). Please implement __stack__()", RuntimeWarning, stacklevel=2)
                result = v.__concat__(expanded_values, dim.name, **kwargs)
                if result is not NotImplemented:
                    assert isinstance(result, Shapable), "__concat__ must return a Shapable object"
                    return result
        # --- else maybe all values are native scalars ---
        from ._tensors import wrap
        try:
            values = tuple([wrap(v) for v in values])
        except ValueError:
            raise MagicNotImplemented(f"At least one item in values must be Shapable but got types {[type(v) for v in values]}")
        return values[0].__stack__(values, dim, **kwargs)
    else:  # multi-dim stack
        assert dim.volume == len(values), f"When passing multiple stack dims, their volume must equal len(values) but got {dim} for {len(values)} values"
        if isinstance(values, dict):
            warnings.warn(f"When stacking a dict along multiple dimensions, the key names are discarded. Got keys {tuple(values.keys())}", RuntimeWarning, stacklevel=2)
            values = tuple(values.values())
        # --- if any value implements Shapable, use stack and unpack_dim ---
        for v in values:
            if hasattr(v, '__stack__') and hasattr(v, '__unpack_dim__'):
                stack_dim = batch('_stack')
                stacked = v.__stack__(values, stack_dim, **kwargs)
                if stacked is not NotImplemented:
                    assert isinstance(stacked, Shapable), "__stack__ must return a Shapable object"
                    assert hasattr(stacked, '__unpack_dim__'), "If a value supports __unpack_dim__, the result of __stack__ must also support it."
                    reshaped = stacked.__unpack_dim__(stack_dim.name, dim, **kwargs)
                    if reshaped is NotImplemented:
                        warnings.warn("__unpack_dim__ is overridden but returned NotImplemented during multi-dimensional stack. This results in unnecessary stack operations.", RuntimeWarning, stacklevel=2)
                    else:
                        return reshaped
        # --- Fallback: multi-level stack ---
        for dim_ in reversed(dim):
            values = [stack(values[i:i + dim_.size], dim_, **kwargs) for i in range(0, len(values), dim_.size)]
        return values[0]


def concat(values: Sequence[PhiTreeNodeType], dim: Union[str, Shape], expand_values=False, **kwargs) -> PhiTreeNodeType:
    """
    Concatenates a sequence of `phiml.math.magic.Shapable` objects, e.g. `Tensor`, along one dimension.
    All values must have the same spatial, instance and channel dims and their sizes must be equal, except for `dim`.
    Batch dims will be added as needed.

    Args:
        values: Tuple or list of `phiml.math.magic.Shapable`, such as `phiml.math.Tensor`
        dim: Concatenation dimension, must be present in all `values`.
            The size along `dim` is determined from `values` and can be set to undefined (`None`).
            Alternatively, a `str` of the form `'t->name:t'` can be specified, where `t` is on of `b d i s c` denoting the dimension type.
            This first packs all dims of the input into a new dim with given name and type, then concatenates the values along this dim.
        expand_values: If `True`, will first add missing dims to all values, not just batch dimensions.
            This allows tensors with different dims to be concatenated.
            The resulting tensor will have all dims that are present in `values`.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dims to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dims must always work without keyword arguments.

    Returns:
        Concatenated `Tensor`

    Examples:
        >>> concat([math.zeros(batch(b=10)), math.ones(batch(b=10))], 'b')
        (bᵇ=20) 0.500 ± 0.500 (0e+00...1e+00)

        >>> concat([vec(x=1, y=0), vec(z=2.)], 'vector')
        (x=1.000, y=0.000, z=2.000) float64
    """
    assert len(values) > 0, f"concat() got empty sequence {values}"
    if isinstance(dim, SHAPE_TYPES):
        dim = dim.name
    assert isinstance(dim, str), f"dim must be a str or Shape but got '{dim}' of type {type(dim)}"
    if '->' in dim:
        dim_type, dim = [s.strip() for s in dim.split('->', 1)]
        dim_type = DIM_FUNCTIONS[INV_CHAR[dim_type]]
        dim = auto(dim, dim_type)
        values = [pack_dims(v, dim_type, dim) for v in values]
        dim = dim.name
    else:
        dim = auto(dim, channel).name
    # --- Filter 0-length values ---
    shapes = [shape(v) for v in values]
    def is_non_zero(s: Shape):
        if dim not in s:
            return True
        size = s.get_size(dim)
        if isinstance(size, int):
            return size > 0
        return True
    filtered_values = [v for v, s in zip(values, shapes) if is_non_zero(s)]
    if not filtered_values:
        return values[0]
    values = filtered_values
    if len(values) == 1:
        return values[0]
    shapes = [s for s in shapes if is_non_zero(s)]
    #  --- Add missing dimensions ---
    if expand_values:
        all_dims = merge_shapes(*shapes, allow_varying_sizes=True)
        all_dims = all_dims.with_dim_size(dim, 1, keep_labels=False)
        values = [expand(v, all_dims - s) for v, s in zip(values, shapes)]
    else:
        for v, s in zip(values, shapes):
            assert dim in s, f"concat dim '{dim}' must be present in the shapes of all values bot got value {type(v).__name__} with shape {s}"
        for v in values[1:]:
            assert set(non_batch(v).names) == set(non_batch(values[0]).names), f"Concatenated values must have the same non-batch dims but got {non_batch(values[0])} and {non_batch(v)}"
        all_batch_dims = merge_shapes(*[s.batch - dim for s in shapes])
        values = [expand(v, all_batch_dims) for v in values]
    # --- First try __concat__ ---
    for v in values:
        if isinstance(v, Shapable):
            if hasattr(v, '__concat__'):
                result = v.__concat__(values, dim, **kwargs)
                if result is not NotImplemented:
                    assert isinstance(result, Shapable), f"__concat__ must return a Shapable object but got {type(result).__name__} from {type(v).__name__} {v}"
                    return result
    # --- Next: try concat attributes for tree nodes ---
    if all(isinstance(v, PhiTreeNode) for v in values):
        attributes = all_attributes(values[0])
        if attributes and all(all_attributes(v) == attributes for v in values):
            new_attrs = {}
            for a in attributes:
                common_shape = merge_shapes(*[shape(getattr(v, a)).without(dim) for v in values])
                a_values = [expand(getattr(v, a), common_shape & shape(v).only(dim)) for v in values]  # expand by dim if missing, and dims of others
                new_attrs[a] = concat(a_values, dim, expand_values=expand_values, **kwargs)
            return copy_with(values[0], **new_attrs)
        else:
            warnings.warn(f"Failed to concat values using value attributes because attributes differ among values {values}")
    # --- Fallback: slice and stack ---
    try:
        unstacked = sum([unstack(v, dim) for v in values], ())
    except MagicNotImplemented:
        raise MagicNotImplemented(f"concat: No value implemented __concat__ and not all values were Sliceable along {dim}. values = {[type(v) for v in values]}")
    if len(unstacked) > 8:
        warnings.warn(f"concat() default implementation is slow on large dims ({dim}={len(unstacked)}). Please implement __concat__()", RuntimeWarning, stacklevel=2)
    dim = shapes[0][dim].with_size(None)
    try:
        return stack(unstacked, dim, **kwargs)
    except MagicNotImplemented:
        raise MagicNotImplemented(f"concat: No value implemented __concat__ and slices could not be stacked. values = {[type(v) for v in values]}")


def ncat(values: Sequence[PhiTreeNodeType], dim: Shape, expand_values=False) -> PhiTreeNodeType:
    """
    Concatenate named components along `dim`.

    Args:
        values: Each value can contain multiple components of `dim` if `dim` is present in its shape.
            Else, it is interpreted as a single component whose name will be determined from the leftover labels of `dim`.
        dim: Single dimension that has labels matching components of `values`.
        expand_values: If `True`, will add all missing dims to values, not just batch dimensions.
            This allows tensors with different dims to be concatenated.
            The resulting tensor will have all dims that are present in `values`.
            If `False`, this may return a non-numeric object instead.

    Returns:
        Same type as any value from `values`.
    """
    order = dim.labels[0]
    assert dim.rank == 1 and order, f"dim needs to be a single dimension with labels but got {dim}"
    named = {}
    unnamed = []
    for value in values:
        s = shape(value)
        if dim in s:
            for n in s[dim].labels[0]:
                named[n] = value[{dim.name: n}]
        else:
            unnamed.append(value)
    missing = [n for n in order if n not in named]
    assert len(missing) == len(unnamed), f"Components do not match dim {dim}. Given: {len(unnamed)} for remaining names {missing}"
    named.update({n: v for v, n in zip(unnamed, missing)})
    components = [named[n] for n in order]
    return stack(components, dim, expand_values=expand_values)


def tcat(values: Sequence[PhiTreeNodeType], dim_type: Callable, expand_values=False, default_name='tcat') -> PhiTreeNodeType:
    """
    Concatenate values by dim type.
    This function first packs all dims of `dim_type` into one dim, then concatenates all `values`.
    Values that do not have a dim of `dim_type` are considered a size-1 slice.

    The name of the first matching dim of `dim_type` is used as the concatenated output dim name.
    If no value has a matching dim, `default_name` is used instead.

    Args:
        values: Values to be concatenated.
        dim_type: Dimension type along which to concatenate.
        expand_values: Whether to add missing other non-batch dims to values as needed.
        default_name: Concatenation dim name if none of the values have a matching dim.

    Returns:
        Same type as any value.
    """
    dims = [dim_type(v) for v in values]
    present_names = tuple(set(sum([s.names for s in dims], ())))
    if len(present_names) == 1:
        dim_name = present_names[0]
    elif len(present_names) > 1:
        if default_name in present_names:
            dim_name = default_name
        else:
            dim_name = present_names[0]
    else:
        dim_name = default_name
    single = dim_type(**{dim_name: 1})
    flat_values = [pack_dims(v, dim_type, dim_type(dim_name)) if s else expand(v, single) for v, s in zip(values, dims)]
    return concat(flat_values, dim_name, expand_values=expand_values)


ccat = partial(tcat, dim_type=channel, expand_values=True, default_name='ccat')
ccat.__doc__ = "Concatenate values along their channel dim, see `tcat`."
icat = partial(tcat, dim_type=instance, expand_values=True, default_name='icat')
icat.__doc__ = "Concatenate values along their instance dim, see `tcat`."
dcat = partial(tcat, dim_type=dual, expand_values=True, default_name='dcat')
dcat.__doc__ = "Concatenate values along their dual dim, see `tcat`."
scat = partial(tcat, dim_type=spatial, expand_values=True, default_name='scat')
scat.__doc__ = "Concatenate values along their spatial dim, see `tcat`."


def expand(value, *dims: Union[Shape, str], **kwargs):
    """
    Adds dims to a `Tensor` or tensor-like object by implicitly repeating the tensor values along the new dimensions.
    If `value` already contains any of the new dimensions, a size and type check is performed for these instead.

    If any of `dims` varies along a dimension that is present neither in `value` nor on `dims`, it will also be added to `value`.

    This function replaces the usual `tile` / `repeat` functions of
    [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.tile.html),
    [PyTorch](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.repeat),
    [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/tile) and
    [Jax](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tile.html).

    Additionally, it replaces the traditional `unsqueeze` / `expand_dims` functions.

    Args:
        value: `phiml.math.magic.Shapable`, such as `phiml.math.Tensor`
            For tree nodes, expands all value attributes by `dims` or the first variable attribute if no value attributes are set.
        *dims: Dimensions to be added as `Shape`
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dims to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dims must always work without keyword arguments.

    Returns:
        Same type as `value`.
    """
    if not dims:
        return value
    dims = concat_shapes_(*[d if isinstance(d, SHAPE_TYPES) else parse_shape_spec(d) for d in dims])
    combined = merge_shapes(value, dims)  # check that existing sizes match
    if not dims.without(shape(value)):  # no new dims to add
        if set(dims) == set(shape(value).only(dims)):  # sizes and labels might differ, though
            return value
    dims &= combined.non_uniform_shape  # add missing non-uniform dims
    # --- First try __expand__
    if hasattr(value, '__expand__'):
        result = value.__expand__(dims, **kwargs)
        if result is not NotImplemented:
            return result
    # --- Next try Tree Node ---
    if isinstance(value, PhiTreeNode):
        new_attributes = {a: expand(getattr(value, a), dims, **kwargs) for a in all_attributes(value)}
        return copy_with(value, **new_attributes)
    # --- Fallback: stack ---
    if hasattr(value, '__stack__'):
        if dims.volume > 8:
            warnings.warn(f"expand() default implementation is slow on large shapes {dims}. Please implement __expand__() for {type(value).__name__} as defined in phiml.math.magic", RuntimeWarning, stacklevel=2)
        for dim in reversed(dims):
            value = stack((value,) * dim.size, dim, **kwargs)
            assert value is not NotImplemented, "Value must implement either __expand__ or __stack__"
        return value
    try:  # value may be a native scalar
        from ._tensors import expand_tensor, wrap
        value = wrap(value)
    except ValueError:
        raise AssertionError(f"Cannot expand non-shapable object {type(value)}")
    return expand_tensor(value, dims)


expand_ = expand


def squeeze(x: PhiTreeNodeType, dims: DimFilter) -> PhiTreeNodeType:
    """
    Remove specific singleton (volume=1) dims from `x`.

    Args:
        x: Tensor or composite type / tree.
        dims: Singleton dims to remove.

    Returns:
        Same type as `x`.
    """
    dims = shape(x).only(dims)
    if not dims:
        return x
    assert dims.volume == 1, f"Cannot squeeze non-singleton dims {dims} from {x}"
    return x[{d: 0 for d in dims.names}]


def rename_dims(value: PhiTreeNodeType,
                dims: DimFilter,
                names: DimFilter,
                **kwargs) -> PhiTreeNodeType:
    """
    Change the name and optionally the type of some dims of `value`.

    Dimensions that are not present on value will be ignored. The corresponding new dims given by `names` will not be added.

    Args:
        value: `Shape` or `Tensor` or `Shapable`.
        dims: Existing dims of `value` as comma-separated `str`, `tuple`, `list`, `Shape` or filter function.
        names: Either

            * Sequence of names matching `dims` as `tuple`, `list` or `str`. This replaces only the dimension names but leaves the types untouched.
            * `Shape` matching `dims` to replace names and types.
            * Dimension type function to replace only types.

        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dims to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dims must always work without keyword arguments.

    Returns:
        Same type as `value`.
    """
    if isinstance(value, SHAPE_TYPES):
        return replace_dims(value, dims, names)
    elif isinstance(value, (Number, bool)):
        return value
    if DEBUG_CHECKS:
        assert isinstance(value, Shapable) and isinstance(value, Shaped), f"value must be a Shape or Shapable but got {type(value).__name__}"
    old_dims, new_dims = _shape_replace(shape(value), dims, names)
    if not new_dims:
        return value
    if new_dims.names == old_dims.names and new_dims == old_dims:
        return value
    # --- First try __replace_dims__ ---
    if hasattr(value, '__replace_dims__'):
        result = value.__replace_dims__(old_dims.names, new_dims, **kwargs)
        if result is not NotImplemented:
            return result
    # --- Next try Tree Node ---
    if isinstance(value, PhiTreeNode):
        return tree_map(rename_dims, value, all_attributes, treat_layout_as_leaf=True, dims=old_dims, names=new_dims, **kwargs)
    # --- Fallback: unstack and stack ---
    if shape(value).only(old_dims).volume > 8:
        warnings.warn(f"rename_dims() default implementation is slow on large dims ({old_dims}). Please implement __replace_dims__() for {type(value).__name__} as defined in phiml.math.magic", RuntimeWarning, stacklevel=2)
    for old_name, new_dim in zip(old_dims.names, new_dims):
        value = stack(unstack(value, old_name), new_dim, **kwargs)
    return value


def b2i(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *batch* dims of `value` to *instance* dimensions. See `rename_dims`. """
    return rename_dims(value, batch, instance)


def c2b(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *channel* dims of `value` to *batch* dimensions. See `rename_dims`. """
    return rename_dims(value, channel, batch)


def s2b(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *spatial* dims of `value` to *batch* dimensions. See `rename_dims`. """
    return rename_dims(value, spatial, batch)


def si2d(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *spatial* and *instance* dims of `value` to *dual* dimensions. See `rename_dims`. """
    return rename_dims(value, lambda s: s.non_channel.non_dual.non_batch, dual)


def c2d(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *channel* dims of `value` to *dual* dimensions. See `rename_dims`. """
    return rename_dims(value, channel, dual)


def p2d(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *primal* dims (instance, spatial, channel) of `value` to *dual* dimensions. See `rename_dims`. """
    return rename_dims(value, primal, dual)


def i2b(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *instance* dims of `value` to *batch* dimensions. See `rename_dims`. """
    return rename_dims(value, instance, batch)


def d2i(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *dual* dims of `value` to *instance* dimensions. See `rename_dims`. """
    return rename_dims(value, dual, instance)


def d2s(value: PhiTreeNodeType) -> PhiTreeNodeType:
    """ Change the type of all *dual* dims of `value` to *spatial* dimensions. See `rename_dims`. """
    return rename_dims(value, dual, spatial)


def pack_dims(value, dims: DimFilter, packed_dim: Union[Shape, str], pos: Optional[int] = None, **kwargs):
    """
    Compresses multiple dims into a single dimension by concatenating the elements.
    Elements along the new dims are laid out according to the order of `dims`.
    If the order of `dims` differs from the current dimension order, the tensor is transposed accordingly.
    This function replaces the traditional `reshape` for these cases.

    The type of the new dimension will be equal to the types of `dims`.
    If `dims` have varying types, the new dimension will be a batch dimension.

    If none of `dims` exist on `value`, `packed_dim` will be added only if it is given with a definite size and `value` is not a primitive type.

    See Also:
        `unpack_dim()`

    Args:
        value: `phiml.math.magic.Shapable`, such as `phiml.math.Tensor`.
        dims: Dimensions to be compressed in the specified order.
        packed_dim: Single-dimension `Shape`.
        pos: Index of new dimension. `None` for automatic, `-1` for last, `0` for first.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dims to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dims must always work without keyword arguments.

    Returns:
        Same type as `value`.

    Examples:
        >>> pack_dims(math.zeros(spatial(x=4, y=3)), spatial, instance('points'))
        (pointsⁱ=12) const 0.0
    """
    if isinstance(value, (Number, bool)):
        return value
    if DEBUG_CHECKS:
        assert isinstance(value, Shapable) and isinstance(value, Sliceable) and isinstance(value, Shaped), f"value must be Shapable but got {type(value)}"
    packed_dim = auto(packed_dim, dims if callable(dims) else None) if isinstance(packed_dim, str) else packed_dim
    dims = shape(value).only(dims, reorder=True)
    if packed_dim in shape(value):
        assert packed_dim in dims, f"Cannot pack dims into new dimension {packed_dim} because it already exists on value {value} and is not packed."
    if len(dims) == 0 or all(dim not in shape(value) for dim in dims):
        return value if packed_dim.size is None else expand(value, packed_dim, **kwargs)  # Inserting size=1 can cause shape errors
    elif len(dims) == 1 and packed_dim.rank == 1:
        return rename_dims(value, dims, packed_dim, **kwargs)
    elif len(dims) == 1 and packed_dim.rank > 1:
        return unpack_dim(value, dims, packed_dim, **kwargs)
    # --- First try __pack_dims__ ---
    if hasattr(value, '__pack_dims__'):
        result = value.__pack_dims__(dims, packed_dim, pos, **kwargs)
        if result is not NotImplemented:
            return result
    # --- Next try Tree Node ---
    if isinstance(value, PhiTreeNode):
        return tree_map(pack_dims, value, attr_type=all_attributes, dims=dims, packed_dim=packed_dim, pos=pos, **kwargs)
    # --- Fallback: unstack and stack ---
    if shape(value).only(dims).volume > 8:
        warnings.warn(f"pack_dims() default implementation is slow on large dims ({shape(value).only(dims)}). Please implement __pack_dims__() for {type(value).__name__} as defined in phiml.math.magic", RuntimeWarning, stacklevel=2)
    return stack(unstack(value, dims), packed_dim, **kwargs)


def dpack(value, packed_dim: Union[Shape, str], pos: Optional[int] = None, **kwargs):
    """Short for `pack_dims(..., dims=dual)"""
    return pack_dims(value, dual, packed_dim, pos=pos, **kwargs)


def spack(value, packed_dim: Union[Shape, str], pos: Optional[int] = None, **kwargs):
    """Short for `pack_dims(..., dims=spatial)"""
    return pack_dims(value, spatial, packed_dim, pos=pos, **kwargs)


def ipack(value, packed_dim: Union[Shape, str], pos: Optional[int] = None, **kwargs):
    """Short for `pack_dims(..., dims=instance)"""
    return pack_dims(value, instance, packed_dim, pos=pos, **kwargs)


def cpack(value, packed_dim: Union[Shape, str], pos: Optional[int] = None, **kwargs):
    """Short for `pack_dims(..., dims=channel)"""
    return pack_dims(value, channel, packed_dim, pos=pos, **kwargs)


def unpack_dim(value, dim: DimFilter, *unpacked_dims: Union[Shape, Sequence[Shape]], **kwargs):
    """
    Decompresses a dimension by unstacking the elements along it.
    This function replaces the traditional `reshape` for these cases.
    The compressed dimension `dim` is assumed to contain elements laid out according to the order of `unpacked_dims`.

    If `dim` does not exist on `value`, this function will return `value` as-is. This includes primitive types.

    See Also:
        `pack_dims()`

    Args:
        value: `phiml.math.magic.Shapable`, such as `Tensor`, for which one dimension should be split.
        dim: Single dimension to be decompressed.
        *unpacked_dims: Either vararg `Shape`, ordered dims to replace `dim`, fulfilling `unpacked_dims.volume == shape(self)[dim].rank`.
            This results in a single tensor output.
            Alternatively, pass a `tuple` or `list` of shapes to unpack a dim into multiple tensors whose combined volumes match `dim.size`.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dims to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dims must always work without keyword arguments.

    Returns:
        Same type as `value`.

    Examples:
        >>> unpack_dim(math.zeros(instance(points=12)), 'points', spatial(x=4, y=3))
        (xˢ=4, yˢ=3) const 0.0
    """
    if len(unpacked_dims) == 1 and isinstance(unpacked_dims[0], (tuple, list)):
        from ._ops import unflatten_unpack
        return unflatten_unpack(value, dim, unpacked_dims[0])
    if isinstance(value, (Number, bool)):
        return value
    if DEBUG_CHECKS:
        assert isinstance(value, Shapable) and isinstance(value, Sliceable) and isinstance(value, Shaped), f"value must be Shapable but got {type(value)}"
    dim = shape(value).only(dim)
    if dim.is_empty:
        return value  # Nothing to do, maybe expand?
    assert dim.rank == 1, f"unpack_dim requires as single dimension to be unpacked but got {dim}"
    dim = dim.name
    unpacked_dims = concat_shapes_(*unpacked_dims)
    if unpacked_dims.rank == 0:
        return value[{dim: 0}]  # remove dim
    elif unpacked_dims.rank == 1:
        return rename_dims(value, dim, unpacked_dims, **kwargs)
    # --- First try __unpack_dim__
    if hasattr(value, '__unpack_dim__'):
        result = value.__unpack_dim__(dim, unpacked_dims, **kwargs)
        if result is not NotImplemented:
            return result
    # --- Next try Tree Node ---
    if isinstance(value, PhiTreeNode) and all_attributes(value):
        new_attributes = {a: unpack_dim(getattr(value, a), dim, unpacked_dims, **kwargs) for a in all_attributes(value)}
        return copy_with(value, **new_attributes)
    # --- Fallback: unstack and stack ---
    if shape(value).only(dim).volume > 8:
        warnings.warn(f"pack_dims() default implementation is slow on large dims ({shape(value).only(dim)}). Please implement __unpack_dim__() for {type(value).__name__} as defined in phiml.math.magic", RuntimeWarning, stacklevel=2)
    unstacked = unstack(value, dim)
    for dim in reversed(unpacked_dims):
        unstacked = [stack(unstacked[i:i+dim.size], dim, **kwargs) for i in range(0, len(unstacked), dim.size)]
    return unstacked[0]


def flatten(value, flat_dim: Shape = instance('flat'), flatten_batch=False, **kwargs):
    """
    Returns a `Tensor` with the same values as `value` but only a single dimension `flat_dim`.
    The order of the values in memory is not changed.

    Args:
        value: `phiml.math.magic.Shapable`, such as `Tensor`.
            If a non-`phiml.math.magic.Shaped` object or one with an empty `Shape` is passed, it is returned without alteration.
        flat_dim: Dimension name and type as `Shape` object. The size is ignored.
        flatten_batch: Whether to flatten batch dims as well.
            If `False`, batch dims are kept, only onn-batch dims are flattened.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dims to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dims must always work without keyword arguments.

    Returns:
        Same type as `value`.

    Examples:
        >>> flatten(math.zeros(spatial(x=4, y=3)))
        (flatⁱ=12) const 0.0
    """
    assert isinstance(flat_dim, SHAPE_TYPES) and flat_dim.rank == 1, flat_dim
    if not isinstance(value, Shaped):
        return value
    if shape(value).is_empty:
        return value
    assert isinstance(value, Shapable) and isinstance(value, Shaped), f"value must be Shapable but got {type(value)}"
    # --- First try __flatten__ ---
    if hasattr(value, '__flatten__'):
        result = value.__flatten__(flat_dim, flatten_batch, **kwargs)
        if result is not NotImplemented:
            return result
    # There is no tree node implementation for flatten because pack_dims is just as fast
    # --- Fallback: pack_dims ---
    return pack_dims(value, shape(value) if flatten_batch else non_batch(value), flat_dim, **kwargs)


# Other Ops


def cast(x: MagicType, dtype: Union[DType, type]) -> OtherMagicType:
    """
    Casts `x` to a different data type.

    Implementations:

    * NumPy: [`x.astype()`](numpy.ndarray.astype)
    * PyTorch: [`x.to()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to)
    * TensorFlow: [`tf.cast`](https://www.tensorflow.org/api_docs/python/tf/cast)
    * Jax: [`jax.numpy.array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html)

    See Also:
        `to_float`, `to_int32`, `to_int64`, `to_complex`.

    Args:
        x: `Tensor`
        dtype: New data type as `phiml.math.DType`, e.g. `DType(int, 16)`.

    Returns:
        `Tensor` with data type `dtype`
    """
    if not isinstance(dtype, DType):
        dtype = DType.as_dtype(dtype)
    if hasattr(x, '__cast__'):
        return x.__cast__(dtype)
    elif isinstance(x, (Number, bool)):
        return dtype.kind(x)
    elif isinstance(x, PhiTreeNode):
        attrs = {key: getattr(x, key) for key in value_attributes(x)}
        new_attrs = {k: cast(v, dtype) for k, v in attrs.items()}
        return copy_with(x, **new_attrs)
    try:
        backend = choose_backend(x)
        return backend.cast(x, dtype)
    except NoBackendFound:
        if dtype.kind == bool:
            return bool(x)
        raise ValueError(f"Cannot cast object of type '{type(x).__name__}'")


def bool_to_int(x: MagicType, bits=32):
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, Number):
        return x
    if hasattr(x, 'dtype') and isinstance(x.dtype, DType):
        return cast(x, DType(int, bits)) if x.dtype.kind == bool else x
    elif isinstance(x, PhiTreeNode):
        return tree_map(bool_to_int, x, bits=32)
    try:
        backend = choose_backend(x)
        return backend.cast(x, DType(int, bits)) if backend.dtype(x).kind == bool else x
    except NoBackendFound:
        raise ValueError(f"Cannot cast object of type '{type(x).__name__}'")


def _is_data_array(sequence):
    try:
        all([np.asarray(v).dtype != object for v in sequence])
    except ValueError:  # e.g. inhomogeneous
        return False


def _contains_tensor(obj, include_layout=True):
    tensors = []
    tree_map(lambda t: tensors.append(t), obj, all_attributes, treat_layout_as_leaf=include_layout)
    return bool(tensors)
