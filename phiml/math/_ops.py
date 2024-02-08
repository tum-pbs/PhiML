import functools
import math
import warnings
from numbers import Number
from typing import Tuple, Callable, Any, Union, Optional, Dict, Collection, Sequence

import numpy as np

from . import extrapolation as e_
from ._magic_ops import expand, pack_dims, unpack_dim, cast, copy_with, value_attributes, bool_to_int, tree_map, concat, stack, unstack, rename_dims
from ._shape import (Shape, EMPTY_SHAPE,
                     spatial, batch, channel, instance, merge_shapes, parse_dim_order, concat_shapes,
                     IncompatibleShapes, DimFilter, non_batch, dual, non_channel, shape, shape as get_shape)
from ._sparse import CompressedSparseMatrix, dense, SparseCoordinateTensor, get_format, to_format, stored_indices, tensor_like, sparse_dims, same_sparsity_pattern, is_sparse, sparse_dot, sparse_sum, sparse_gather, sparse_max, sparse_min
from ._tensors import (Tensor, wrap, tensor, broadcastable_native_tensors, NativeTensor, TensorStack,
                       custom_op2, compatible_tensor, variable_attributes, disassemble_tree, assemble_tree,
                       is_scalar, Layout, expand_tensor, TensorOrTree, cached)
from ..backend import default_backend, choose_backend, Backend, get_precision, convert as b_convert, BACKENDS, NoBackendFound, ComputeDevice, NUMPY
from ..backend._dtype import DType, combine_types
from .magic import PhiTreeNode


def choose_backend_t(*values, prefer_default=False) -> Backend:
    """
    Choose backend for given `Tensor` or native tensor values.
    Backends need to be registered to be available, e.g. via `init()` or `use()`.

    Args:
        *values: Sequence of `Tensor`s, native tensors or constants.
        prefer_default: Whether to always select the default backend if it can work with `values`, see `default_backend()`.

    Returns:
        The selected `phiml.math.backend.Backend`
    """
    natives = sum([v._natives() if isinstance(v, Tensor) else (v,) for v in values], ())
    return choose_backend(*natives, prefer_default=prefer_default)


def convert(x, backend: Backend = None, use_dlpack=True):
    """
    Convert the native representation of a `Tensor` or `phiml.math.magic.PhiTreeNode` to the native format of `backend`.

    *Warning*: This operation breaks the automatic differentiation chain.

    See Also:
        `phiml.math.backend.convert()`.

    Args:
        x: `Tensor` to convert. If `x` is a `phiml.math.magic.PhiTreeNode`, its variable attributes are converted.
        backend: Target backend. If `None`, uses the current default backend, see `phiml.math.backend.default_backend()`.

    Returns:
        `Tensor` with native representation belonging to `backend`.
    """
    if isinstance(x, Tensor):
        return x._op1(lambda native: b_convert(native, backend, use_dlpack=use_dlpack))
    elif isinstance(x, PhiTreeNode):
        return copy_with(x, **{a: convert(getattr(x, a), backend, use_dlpack=use_dlpack) for a in variable_attributes(x)})
    else:
        return b_convert(x, backend, use_dlpack=use_dlpack)


def to_device(value, device: ComputeDevice or str, convert=True, use_dlpack=True):
    """
    Allocates the tensors of `value` on `device`.
    If the value already exists on that device, this function may either create a copy of `value` or return `value` directly.

    See Also:
        `to_cpu()`.

    Args:
        value: `Tensor` or `phiml.math.magic.PhiTreeNode` or native tensor.
        device: Device to allocate value on.
            Either `ComputeDevice` or category `str`, such as `'CPU'` or `'GPU'`.
        convert: Whether to convert tensors that do not belong to the corresponding backend to compatible native tensors.
            If `False`, this function has no effect on numpy tensors.
        use_dlpack: Only if `convert==True`.
            Whether to use the DLPack library to convert from one GPU-enabled backend to another.

    Returns:
        Same type as `value`.
    """
    assert isinstance(device, (ComputeDevice, str)), f"device must be a ComputeDevice or str but got {type(device)}"
    return tree_map(_to_device, value, device=device, convert_to_backend=convert, use_dlpack=use_dlpack)


def _to_device(value: Tensor or Any, device: ComputeDevice or str, convert_to_backend: bool, use_dlpack: bool):
    if isinstance(value, Tensor):
        if not convert and value.default_backend == NUMPY:
            return value
        natives = [_to_device(n, device, convert_to_backend, use_dlpack) for n in value._natives()]
        return value._with_natives_replaced(natives)
    else:
        old_backend = choose_backend(value)
        if isinstance(device, str):
            device = old_backend.list_devices(device)[0]
        if old_backend != device.backend:
            if convert_to_backend:
                value = b_convert(value, device.backend, use_dlpack=use_dlpack)
            else:
                return value
        return device.backend.allocate_on_device(value, device)


def all_available(*values) -> bool:
    """
    Tests if all tensors contained in the given `values` are currently known and can be read.
    Placeholder tensors used to trace functions for just-in-time compilation or matrix construction are considered not available, even when they hold example values like with PyTorch's JIT.

    Tensors are not available during `jit_compile()`, `jit_compile_linear()` or while using TensorFlow's legacy graph mode.
    
    Tensors are typically available when the backend operates in eager mode and is not currently tracing a function.

    This can be used instead of the native checks

    * PyTorch: `torch._C._get_tracing_state()`
    * TensorFlow: `tf.executing_eagerly()`
    * Jax: `isinstance(x, jax.core.Tracer)`

    Args:
        values: Tensors to check.

    Returns:
        `True` if no value is a placeholder or being traced, `False` otherwise.
    """
    _, tensors = disassemble_tree(values, cache=False)
    return all([t.available for t in tensors])


def seed(seed: int):
    """
    Sets the current seed of all backends and the built-in `random` package.

    Calling this function with a fixed value at the start of an application yields reproducible results
    as long as the same backend is used.

    Args:
        seed: Seed to use.
    """
    for backend in BACKENDS:
        backend.seed(seed)
    import random
    random.seed(0)


def native(value: Union[Tensor, Number, tuple, list, Any]):
    """
    Returns the native tensor representation of `value`.
    If `value` is a `phiml.math.Tensor`, this is equal to calling `phiml.math.Tensor.native()`.
    Otherwise, checks that `value` is a valid tensor object and returns it.

    Args:
        value: `Tensor` or native tensor or tensor-like.

    Returns:
        Native tensor representation

    Raises:
        ValueError if the tensor cannot be transposed to match target_shape
    """
    if isinstance(value, Tensor):
        return value.native()
    else:
        choose_backend(value)  # check that value is a native tensor
        return value


def numpy(value: Union[Tensor, Number, tuple, list, Any]):
    """
    Converts `value` to a `numpy.ndarray` where value must be a `Tensor`, backend tensor or tensor-like.
    If `value` is a `phiml.math.Tensor`, this is equal to calling `phiml.math.Tensor.numpy()`.

    *Note*: Using this function breaks the autograd chain. The returned tensor is not differentiable.
    To get a differentiable tensor, use `Tensor.native()` instead.

    Transposes the underlying tensor to match the name order and adds singleton dimensions for new dimension names.
    If a dimension of the tensor is not listed in `order`, a `ValueError` is raised.

    If `value` is a NumPy array, it may be returned directly.

    Returns:
        NumPy representation of `value`

    Raises:
        ValueError if the tensor cannot be transposed to match target_shape
    """
    if isinstance(value, Tensor):
        return value.numpy()
    else:
        backend = choose_backend(value)
        return backend.numpy(value)


def reshaped_native(value: Tensor,
                    groups: Union[tuple, list],
                    force_expand: Any = True,
                    to_numpy=False):
    """
    Returns a native representation of `value` where dimensions are laid out according to `groups`.

    See Also:
        `native()`, `pack_dims()`, `reshaped_tensor()`, `reshaped_numpy()`.

    Args:
        value: `Tensor`
        groups: `tuple` or `list` of dimensions to be packed into one native dimension. Each entry must be one of the following:

            * `str`: the name of one dimension that is present on `value`.
            * `Shape`: Dimensions to be packed. If `force_expand`, missing dimensions are first added, otherwise they are ignored.
            * Filter function: Packs all dimensions of this type that are present on `value`.
            * Ellipsis `...`: Packs all remaining dimensions into this slot. Can only be passed once.

        force_expand: `bool` or sequence of dimensions.
            If `True`, repeats the tensor along missing dimensions.
            If `False`, puts singleton dimensions where possible.
            If a sequence of dimensions is provided, only forces the expansion for groups containing those dimensions.
        to_numpy: If True, converts the native tensor to a `numpy.ndarray`.

    Returns:
        Native tensor with dimensions matching `groups`.
    """
    assert isinstance(value, Tensor), f"value must be a Tensor but got {type(value)}"
    assert value.shape.is_uniform, f"Only uniform (homogenous) tensors can be converted to native but got shape {value.shape}"
    assert isinstance(groups, (tuple, list)), f"groups must be a tuple or list but got {type(value)}"
    order = []
    if Ellipsis in groups:
        ellipsis_dims = value.shape.without([g for g in groups if g is not Ellipsis])
        groups = [ellipsis_dims if g is Ellipsis else g for g in groups]
    groups = [group(value) if callable(group) else group for group in groups]
    for i, group in enumerate(groups):
        if isinstance(group, Shape):
            present = value.shape.only(group)
            if force_expand is True or present.volume > 1 or (force_expand is not False and group.only(force_expand).volume > 1):
                value = expand(value, group)
            value = pack_dims(value, group, batch(f"group{i}"))
            order.append(f"group{i}")
        else:
            assert isinstance(group, str), f"Groups must be either single-dim str or Shape but got {group}"
            assert ',' not in group, f"When packing multiple dimensions, pass a well-defined Shape instead of a comma-separated str. Got {group}"
            order.append(group)
    return value.numpy(order) if to_numpy else value.native(order)


def reshaped_numpy(value: Tensor, groups: Union[tuple, list], force_expand: Any = True):
    """
    Returns the NumPy representation of `value` where dimensions are laid out according to `groups`.

    See Also:
        `numpy()`, `reshaped_native()`, `pack_dims()`, `reshaped_tensor()`.

    Args:
        value: `Tensor`
        groups: Sequence of dimension names as `str` or groups of dimensions to be packed_dim as `Shape`.
        force_expand: `bool` or sequence of dimensions.
            If `True`, repeats the tensor along missing dimensions.
            If `False`, puts singleton dimensions where possible.
            If a sequence of dimensions is provided, only forces the expansion for groups containing those dimensions.

    Returns:
        NumPy `ndarray` with dimensions matching `groups`.
    """
    return reshaped_native(value, groups, force_expand=force_expand, to_numpy=True)


def reshaped_tensor(value: Any,
                    groups: Union[tuple, list],
                    check_sizes=False,
                    convert=True):
    """
    Creates a `Tensor` from a native tensor or tensor-like whereby the dimensions of `value` are split according to `groups`.

    See Also:
        `phiml.math.tensor()`, `reshaped_native()`, `unpack_dim()`.

    Args:
        value: Native tensor or tensor-like.
        groups: Sequence of dimension groups to be packed_dim as `tuple[Shape]` or `list[Shape]`.
        check_sizes: If True, group sizes must match the sizes of `value` exactly. Otherwise, allows singleton dimensions.
        convert: If True, converts the data to the native format of the current default backend.
            If False, wraps the data in a `Tensor` but keeps the given data reference if possible.

    Returns:
        `Tensor` with all dimensions from `groups`
    """
    assert all(isinstance(g, Shape) for g in groups), "groups must be a sequence of Shapes"
    dims = [batch(f'group{i}') for i, group in enumerate(groups)]
    try:
        value = tensor(value, *dims, convert=convert)
    except IncompatibleShapes:
        raise IncompatibleShapes(f"Cannot reshape native tensor {type(value)} with sizes {value.shape} given groups {groups}")
    for i, group in enumerate(groups):
        if value.shape.get_size(f'group{i}') == group.volume:
            value = unpack_dim(value, f'group{i}', group)
        elif check_sizes:
            raise AssertionError(f"Group {group} does not match dimension {i} of value {value.shape}")
        else:
            value = unpack_dim(value, f'group{i}', group)
    return value


def copy(value: Tensor):
    """
    Copies the data buffer and encapsulating `Tensor` object.

    Args:
        value: `Tensor` to be copied.

    Returns:
        Copy of `value`.
    """
    if value._is_tracer:
        warnings.warn("Tracing tensors cannot be copied.", RuntimeWarning)
        return value
    return value._op1(lambda native: choose_backend(native).copy(native))


def native_call(f: Callable, *inputs: Tensor, channels_last=None, channel_dim='vector', spatial_dim=None):
    """
    Calls `f` with the native representations of the `inputs` tensors in standard layout and returns the result as a `Tensor`.

    All inputs are converted to native tensors (including precision cast) depending on `channels_last`:

    * `channels_last=True`: Dimension layout `(total_batch_size, spatial_dims..., total_channel_size)`
    * `channels_last=False`: Dimension layout `(total_batch_size, total_channel_size, spatial_dims...)`

    All batch dimensions are compressed into a single dimension with `total_batch_size = input.shape.batch.volume`.
    The same is done for all channel dimensions.

    Additionally, missing batch and spatial dimensions are added so that all `inputs` have the same batch and spatial shape.

    Args:
        f: Function to be called on native tensors of `inputs`.
            The function output must have the same dimension layout as the inputs, unless overridden by `spatial_dim`,
            and the batch size must be identical.
        *inputs: Uniform `Tensor` arguments
        channels_last: (Optional) Whether to put channels as the last dimension of the native representation.
            If `None`, the channels are put in the default position associated with the current backend,
            see `phiml.math.backend.Backend.prefers_channels_last()`.
        channel_dim: Name of the channel dimension of the result.
        spatial_dim: Name of the spatial dimension of the result.

    Returns:
        `Tensor` with batch and spatial dimensions of `inputs`, unless overridden by `spatial_dim`,
        and single channel dimension `channel_dim`.
    """
    if channels_last is None:
        try:
            backend = choose_backend(f)
        except NoBackendFound:
            backend = choose_backend_t(*inputs, prefer_default=True)
        channels_last = backend.prefers_channels_last()
    batch = merge_shapes(*[i.shape.batch for i in inputs])
    spatial = merge_shapes(*[i.shape.spatial for i in inputs])
    natives = []
    for i in inputs:
        groups = (batch, *i.shape.spatial.names, i.shape.channel) if channels_last else (batch, i.shape.channel, *i.shape.spatial.names)
        natives.append(reshaped_native(i, groups, force_expand=False))
    output = f(*natives)
    if isinstance(channel_dim, str):
        channel_dim = channel(channel_dim)
    assert isinstance(channel_dim, Shape), "channel_dim must be a Shape or str"
    if isinstance(output, (tuple, list)):
        raise NotImplementedError()
    else:
        if spatial_dim is None:
            groups = (batch, *spatial, channel_dim) if channels_last else (batch, channel_dim, *spatial)
        else:
            if isinstance(spatial_dim, str):
                spatial_dim = spatial(spatial_dim)
            assert isinstance(spatial_dim, Shape), "spatial_dim must be a Shape or str"
            groups = (batch, *spatial_dim, channel_dim) if channels_last else (batch, channel_dim, *spatial_dim)
        result = reshaped_tensor(output, groups, convert=False)
        if result.shape.get_size(channel_dim.name) == 1 and not channel_dim.item_names[0]:
            result = result.dimension(channel_dim.name)[0]  # remove vector dim if not required
        return result


def print_(obj: Union[Tensor, PhiTreeNode, Number, tuple, list, None] = None, name: str = ""):
    """
    Print a tensor with no more than two spatial dimensions, slicing it along all batch and channel dimensions.
    
    Unlike NumPy's array printing, the dimensions are sorted.
    Elements along the alphabetically first dimension is printed to the right, the second dimension upward.
    Typically, this means x right, y up.

    Args:
        obj: tensor-like
        name: name of the tensor

    Returns:

    """
    def variables(obj) -> dict:
        if hasattr(obj, '__variable_attrs__') or hasattr(obj, '__value_attrs__'):
            return {f".{a}": getattr(obj, a) for a in variable_attributes(obj)}
        elif isinstance(obj, (tuple, list)):
            return {f"[{i}]": item for i, item in enumerate(obj)}
        elif isinstance(obj, dict):
            return obj
        else:
            raise ValueError(f"Not PhiTreeNode: {type(obj)}")

    if name:
        print(" " * 12 + name)
    if obj is None:
        print("None")
    elif isinstance(obj, Tensor):
        print(f"{obj:full}")
    elif isinstance(obj, PhiTreeNode):
        for n, val in variables(obj).items():
            print_(val, name + n)
    else:
        print(f"{wrap(obj):full}")


def slice_off(x, *slices: Dict[str, Union[slice, int, str]]):
    """

    Args:
        x: Any instance of `phiml.math.magic.Shapable`
        *slices:

    Returns:

    """
    if not slices:
        return x
    x_shape = shape(x)
    dims = set().union(*[s.keys() for s in slices])
    dims = x_shape.only(dims).names
    depth = max(len(s) for s in slices)
    if depth == 1:
        if len(dims) == 1:
            d = dims[0]
            if all(all(_edge_slice(x_shape, dim, s) for dim, s in s_dict.items()) for s_dict in slices):  # only edges
                edge_slices = [_edge_slice(x_shape, dim, s) for s_dict in slices for dim, s in s_dict.items()]
                if any(s.start == 0 and s.stop is None for s in edge_slices):  # everything sliced off
                    return x[{d: slice(0, 0)}]
                start_slices = [s for s in edge_slices if s.start == 0]
                end_slices = [s for s in edge_slices if s.stop is None]
                start = max(s.stop for s in start_slices) if start_slices else 0  # at this point, s.stop must be an int
                end = min(s.start for s in end_slices) if end_slices else None
                return x[{d: slice(start, end)}]
            else:
                size = x_shape.get_size(d)
                mask = np.ones(size, dtype=np.bool_)
                for s_dict in slices:
                    s = next(iter(s_dict.values()))
                    if isinstance(s, str):
                        names = x_shape.get_item_names(d)
                        s = [names.index(n.strip()) for n in s.split(',')]
                    mask[s] = 0
                return boolean_mask(x, d, wrap(mask, x_shape[d]))
    unstack_dim = x_shape.only(_preferred_unstack_dim(x, dims))
    x_slices = unstack(x, unstack_dim)
    x_slices_out = []
    for i, x_slice in enumerate(x_slices):
        slices_without_unstack_dim = [{k: v for k, v in s_dict.items() if k != unstack_dim.name} for s_dict in slices if _includes_slice(s_dict, unstack_dim, i)]
        sliced_x_slice = slice_off(x_slice, *slices_without_unstack_dim)
        x_slices_out.append(sliced_x_slice)
    assembled = stack(x_slices_out, unstack_dim)
    slices_for_unstack_dim_only = [s_dict for s_dict in slices if len(s_dict) == 1 and unstack_dim.name in s_dict]
    result = slice_off(assembled, *slices_for_unstack_dim_only)
    return result


def _edge_slice(x_shape: Shape, dim: str, s):
    size = x_shape.get_size(dim)
    if isinstance(s, str):
        s = [n.strip() for n in s.split(',')]
    if isinstance(s, slice):
        reaches_end = s.stop is None or s.stop >= size
        if not s.start:
            return slice(0, None if reaches_end else s.stop)
        elif reaches_end:
            return slice(s.start, None)
    elif isinstance(s, int):
        if s == 0:
            return slice(0, 1)
        elif s == -1 or s == size - 1:
            return slice(size - 1, None)
    elif isinstance(s, (tuple, list)):
        names = x_shape.get_item_names(dim)
        indices = [i if isinstance(i, int) else names.index(i) for i in s]
        if all(n in indices for n in range(max(indices))):
            return slice(0, max(indices) + 1)
        elif all(n in indices for n in range(min(indices), size)):
            return slice(min(indices), None)


def _preferred_unstack_dim(x, dims: Collection[str]) -> str:
    if shape(x).is_non_uniform:
        nu = shape(x).non_uniform
        return nu.shape.without('dims').names[0]
    else:
        return min(dims, key=lambda d: x.shape.get_size(d))


def _includes_slice(s_dict: dict, dim: Shape, i: int):
    if dim.name not in s_dict:
        return True
    s = s_dict[dim.name]
    if isinstance(s, str):
        s = [n.strip() for n in s.split(',')]
    if isinstance(s, int):
        return s == i
    elif isinstance(s, slice):
        return (s.start or 0) <= i < (s.stop if s.stop is not None else dim.size)
    elif isinstance(s, (tuple, list)):
        names = dim.item_names[0]
        indices = [i if isinstance(i, int) else names.index(i) for i in s]
        return i in indices


def _initialize(uniform_initializer, shapes: Tuple[Shape]) -> Tensor:
    shape = concat_shapes(*shapes)
    assert shape.well_defined, f"When creating a Tensor, shape needs to have definitive sizes but got {shape}"
    if shape.is_non_uniform:
        stack_dim = shape.shape.without('dims')[0:1]
        shapes = shape.unstack(stack_dim.name)
        tensors = [_initialize(uniform_initializer, s) for s in shapes]
        return stack_tensors(tensors, stack_dim)
    else:
        return uniform_initializer(shape)


def zeros(*shape: Shape, dtype: Union[DType, tuple, type] = None) -> Tensor:
    """
    Define a tensor with specified shape with value `0.0` / `0` / `False` everywhere.
    
    This method may not immediately allocate the memory to store the values.

    See Also:
        `zeros_like()`, `ones()`.

    Args:
        *shape: This (possibly empty) sequence of `Shape`s is concatenated, preserving the order.
        dtype: Data type as `DType` object. Defaults to `float` matching the current precision setting.

    Returns:
        `Tensor`
    """
    return _initialize(lambda shape: expand_tensor(NativeTensor(default_backend().zeros((), dtype=DType.as_dtype(dtype)), EMPTY_SHAPE), shape), shape)


def zeros_like(obj: Union[Tensor, PhiTreeNode]) -> Union[Tensor, PhiTreeNode]:
    """ Create a `Tensor` containing only `0.0` / `0` / `False` with the same shape and dtype as `obj`. """
    nest, values = disassemble_tree(obj, cache=False)
    zeros_ = []
    for val in values:
        val = wrap(val)
        with val.default_backend:
            zeros_.append(zeros(val.shape, dtype=val.dtype))
    return assemble_tree(nest, zeros_)


def ones(*shape: Shape, dtype: Union[DType, tuple, type] = None) -> Tensor:
    """
    Define a tensor with specified shape with value `1.0`/ `1` / `True` everywhere.
    
    This method may not immediately allocate the memory to store the values.

    See Also:
        `ones_like()`, `zeros()`.

    Args:
        *shape: This (possibly empty) sequence of `Shape`s is concatenated, preserving the order.
        dtype: Data type as `DType` object. Defaults to `float` matching the current precision setting.

    Returns:
        `Tensor`
    """
    return _initialize(lambda shape: expand_tensor(NativeTensor(default_backend().ones((), dtype=DType.as_dtype(dtype)), EMPTY_SHAPE), shape), shape)


def ones_like(value: Tensor) -> Tensor:
    """ Create a `Tensor` containing only `1.0` / `1` / `True` with the same shape and dtype as `obj`. """
    return zeros_like(value) + 1


def random_normal(*shape: Shape, dtype: Union[DType, tuple, type] = None) -> Tensor:
    """
    Creates a `Tensor` with the specified shape, filled with random values sampled from a normal / Gaussian distribution.

    Implementations:

    * NumPy: [`numpy.random.standard_normal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_normal.html)
    * PyTorch: [`torch.randn`](https://pytorch.org/docs/stable/generated/torch.randn.html)
    * TensorFlow: [`tf.random.normal`](https://www.tensorflow.org/api_docs/python/tf/random/normal)
    * Jax: [`jax.random.normal`](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.normal.html)

    Args:
        *shape: This (possibly empty) sequence of `Shape`s is concatenated, preserving the order.
        dtype: (optional) floating point `DType`. If `None`, a float tensor with the current default precision is created, see `get_precision()`.

    Returns:
        `Tensor`
    """

    def uniform_random_normal(shape):
        native = choose_backend(*shape.sizes, prefer_default=True).random_normal(shape.sizes, DType.as_dtype(dtype))
        return NativeTensor(native, shape)

    return _initialize(uniform_random_normal, shape)


def random_uniform(*shape: Shape,
                   low: Union[Tensor, float] = 0,
                   high: Union[Tensor, float] = 1,
                   dtype: Union[DType, tuple, type] = None) -> Tensor:
    """
    Creates a `Tensor` with the specified shape, filled with random values sampled from a uniform distribution.

    Args:
        *shape: This (possibly empty) sequence of `Shape`s is concatenated, preserving the order.
        dtype: (optional) `DType` or `(kind, bits)`.
            The dtype kind must be one of `float`, `int`, `complex`.
            If not specified, a `float` tensor with the current default precision is created, see `get_precision()`.
        low: Minimum value, included.
        high: Maximum value, excluded.
    Returns:
        `Tensor`
    """
    if get_shape(low).volume == 1 and get_shape(high).volume == 1:
        def uniform_random_uniform(shape):
            native = choose_backend(low, high, *shape.sizes, prefer_default=True).random_uniform(shape.sizes, low, high, DType.as_dtype(dtype))
            return NativeTensor(native, shape)
        return _initialize(uniform_random_uniform, shape)
    else:
        def uniform_random_uniform(shape):
            native = choose_backend(*shape.sizes, prefer_default=True).random_uniform(shape.sizes, 0, 1, DType.as_dtype(dtype))
            return NativeTensor(native, shape)
        return _initialize(uniform_random_uniform, shape) * (high - low) + low


def transpose(x, axes):
    """
    Swap the dimension order of `x`.
    This operation is generally not necessary for `Tensor`s because tensors will be reshaped under the hood or when getting the native/numpy representations.
    It can be used to transpose native tensors.

    Implementations:

    * NumPy: [`numpy.transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)
    * PyTorch: [`x.permute`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute)
    * TensorFlow: [`tf.transpose`](https://www.tensorflow.org/api_docs/python/tf/transpose)
    * Jax: [`jax.numpy.transpose`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.transpose.html)

    Args:
        x: `Tensor` or native tensor or `phiml.math.magic.Shapable`.
        axes: `tuple` or `list`

    Returns:
        `Tensor` or native tensor, depending on `x`.
    """
    if isinstance(x, Tensor):
        if x.shape[axes] == x.shape.only(axes):  # order is correct
            return x
        new_shape = x.shape[axes]
        packed = pack_dims(x, new_shape, instance('_t_flat'))
        return unpack_dim(packed, '_t_flat', new_shape)
    else:
        return choose_backend(x).transpose(x, axes)


def cumulative_sum(x: Tensor, dim: DimFilter):
    """
    Performs a cumulative sum of `x` along `dim`.

    Implementations:

    * NumPy: [`cumsum`](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html)
    * PyTorch: [`cumsum`](https://pytorch.org/docs/stable/generated/torch.cumsum.html)
    * TensorFlow: [`cumsum`](https://www.tensorflow.org/api_docs/python/tf/math/cumsum)
    * Jax: [`cumsum`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.cumsum.html)

    Args:
        x: `Tensor`
        dim: Dimension along which to sum, as `str` or `Shape`.

    Returns:
        `Tensor` with the same shape as `x`.
    """
    dim = x.shape.only(dim)
    assert len(dim) == 1, f"dim must be a single dimension but got {dim}"
    native_x = x.native(x.shape)
    native_result = choose_backend(native_x).cumsum(native_x, x.shape.index(dim))
    return NativeTensor(native_result, x.shape)


def fftfreq(resolution: Shape, dx: Union[Tensor, float] = 1, dtype: DType = None):
    """
    Returns the discrete Fourier transform sample frequencies.
    These are the frequencies corresponding to the components of the result of `math.fft` on a tensor of shape `resolution`.

    Args:
        resolution: Grid resolution measured in cells
        dx: Distance between sampling points in real space.
        dtype: Data type of the returned tensor (Default value = None)

    Returns:
        `Tensor` holding the frequencies of the corresponding values computed by math.fft
    """
    assert resolution.spatial and f"resolution must contain at least one spatial dimension"
    k = meshgrid(**{dim: np.fft.fftfreq(int(n)) for dim, n in resolution.spatial._named_sizes})
    k /= dx
    return to_float(k) if dtype is None else cast(k, dtype)


def meshgrid(dims: Union[Callable, Shape] = spatial, stack_dim=channel('vector'), **dimensions: Union[int, Tensor, tuple, list, Any]) -> Tensor:
    """
    Generate a mesh-grid `Tensor` from keyword dimensions.

    Args:
        **dimensions: Mesh-grid dimensions, mapping names to values.
            Values may be `int`, 1D `Tensor` or 1D native tensor.
        dims: Dimension type of mesh-grid dimensions, one of `spatial`, `channel`, `batch`, `instance`.
        stack_dim: Channel dim along which grids are stacked.
            This is optional for 1D mesh-grids. In that case returns a `Tensor` without a stack dim if `None` or an empty `Shape` is passed.

    Returns:
        Mesh-grid `Tensor` with the dimensions of `dims` / `dimensions` and `stack_dim`.

    Examples:
        >>> math.meshgrid(x=2, y=2)
        (xˢ=2, yˢ=2, vectorᶜ=x,y) 0.500 ± 0.500 (0e+00...1e+00)

        >>> math.meshgrid(x=2, y=(-1, 1))
        (xˢ=2, yˢ=2, vectorᶜ=x,y) 0.250 ± 0.829 (-1e+00...1e+00)

        >>> math.meshgrid(x=2, stack_dim=None)
        (0, 1) along xˢ
    """
    assert 'dim_type' not in dimensions, f"dim_type has been renamed to dims"
    assert not stack_dim or stack_dim.name not in dimensions
    if isinstance(dims, Shape):
        assert not dimensions, f"When passing a Shape to meshgrid(), no kwargs are allowed"
        dimensions = {d: s for d, s in zip(dims.names, dims.sizes)}
        grid_shape = dims
        dim_values = [tuple(range(s)) for s in dims.sizes]
    else:
        dim_type = dims
        assert callable(dim_type), f"dims must be a Shape or dimension type but got {dims}"
        dim_values = []
        dim_sizes = []
        for dim, spec in dimensions.items():
            if isinstance(spec, int) or (isinstance(spec, Tensor) and spec.rank == 0 and spec.dtype.kind == int):
                dim_values.append(tuple(range(int(spec))))
                dim_sizes.append(spec)
            elif isinstance(spec, Tensor):
                assert spec.rank == 1, f"Only 1D sequences allowed, got {spec} for dimension '{dim}'."
                dim_values.append(spec.native())
                dim_sizes.append(spec.shape.volume)
            else:
                backend = choose_backend(spec)
                shape = backend.staticshape(spec)
                assert len(shape) == 1, "Only 1D sequences allowed, got {spec} for dimension '{dim}'."
                dim_values.append(spec)
                dim_sizes.append(shape[0])
        grid_shape = dim_type(**{dim: size for dim, size in zip(dimensions.keys(), dim_sizes)})
    backend = choose_backend(*dim_values, prefer_default=True)
    indices_list = backend.meshgrid(*dim_values)
    channels = [NativeTensor(t, grid_shape) for t in indices_list]
    if not stack_dim:
        assert len(channels) == 1, f"meshgrid with multiple dimension requires a valid stack_dim but got {stack_dim}"
        return channels[0]
    if stack_dim.item_names[0] is None:
        stack_dim = stack_dim.with_size(tuple(dimensions.keys()))
    return stack_tensors(channels, stack_dim)


def linspace(start: Union[float, Tensor, tuple, list], stop: Union[float, Tensor, tuple, list], dim: Shape) -> Tensor:
    """
    Returns `number` evenly spaced numbers between `start` and `stop` along `dim`.

    If `dim` contains multiple dimensions, evenly spaces values along each dimension, then stacks the result along a new channel dimension called `vector`.

    See Also:
        `arange()`, `meshgrid()`.

    Args:
        start: First value, `int` or `Tensor`.
        stop: Last value, `int` or `Tensor`.
        dim: Linspace dimension of integer size.
            The size determines how many values to linearly space between `start` and `stop`.
            The values will be laid out along `dim`.

    Returns:
        `Tensor`

    Examples:
        >>> math.linspace(0, 1, spatial(x=5))
        (0.000, 0.250, 0.500, 0.750, 1.000) along xˢ

        >>> math.linspace(0, (-1, 1), spatial(x=3))
        (0.000, 0.000); (-0.500, 0.500); (-1.000, 1.000) (xˢ=3, vectorᶜ=2)
    """
    assert isinstance(dim, Shape), f"dim must be a Shape but got {dim}"
    assert dim.is_uniform, f"dim must be uniform but got {dim}"
    start = wrap(start)
    stop = wrap(stop)
    if dim.rank > 1:
        return meshgrid(dim) / (dim - 1) * (stop - start) + start
    if is_scalar(start) and is_scalar(stop):
        start = start.native()
        stop = stop.native()
        native_linspace = choose_backend(start, stop, prefer_default=True).linspace(start, stop, dim.size)
        return NativeTensor(native_linspace, dim)
    else:
        from ._functional import map_
        return map_(linspace, start, stop, dim=dim)


def arange(dim: Shape, start_or_stop: Union[int, None] = None, stop: Union[int, None] = None, step=1):
    """
    Returns evenly spaced values between `start` and `stop`.
    If only one limit is given, `0` is used for the start.

    See Also:
        `range_tensor()`, `linspace()`, `meshgrid()`.

    Args:
        dim: Dimension name and type as `Shape` object.
            The `size` of `dim` is interpreted as `stop` unless `start_or_stop` is specified.
        start_or_stop: (Optional) `int`. Interpreted as `start` if `stop` is specified as well. Otherwise this is `stop`.
        stop: (Optional) `int`. `stop` value.
        step: Distance between values.

    Returns:
        `Tensor`
    """
    if start_or_stop is None:
        assert stop is None, "start_or_stop must be specified when stop is given."
        assert isinstance(dim.size, int), "When start_or_stop is not specified, dim.size must be an integer."
        start, stop = 0, dim.size
    elif stop is None:
        start, stop = 0, start_or_stop
    else:
        start = start_or_stop
    native = choose_backend(start, stop, prefer_default=True).range(start, stop, step, DType(int, 32))
    return NativeTensor(native, dim.with_sizes(len(native)))


def range_tensor(*shape: Shape):
    """
    Returns a `Tensor` with given `shape` containing the linear indices of each element.
    For 1D tensors, this equivalent to `arange()` with `step=1`.

    See Also:
        `arange()`, `meshgrid()`.

    Args:
        shape: Tensor shape.

    Returns:
        `Tensor`
    """
    shape = concat_shapes(*shape)
    data = arange(spatial('range'), 0, shape.volume)
    return unpack_dim(data, 'range', shape)


def stack_tensors(values: Union[tuple, list], dim: Shape):
    if len(values) == 1 and not dim:
        return values[0]
    values = [wrap(v) for v in values]
    values = cast_same(*values)

    def inner_stack(*values):
        if len(values) > 1 or not isinstance(values[0], NativeTensor):
            if all(isinstance(t, SparseCoordinateTensor) for t in values):
                if all(values[0]._indices is t._indices for t in values):
                    return values[0]._with_values(stack_tensors([v._values for v in values], dim))
            return TensorStack(values, dim)
        else:
            value: NativeTensor = values[0]
            return NativeTensor(value._native, value._native_shape, value.shape & dim.with_size(1))

    result = broadcast_op(inner_stack, values)
    return result


def concat_tensor(values: Union[tuple, list], dim: str) -> Tensor:
    assert len(values) > 0, "concat() got empty sequence"
    assert isinstance(dim, str), f"dim must be a single-dimension Shape but got '{dim}' of type {type(dim)}"
    if any(v._is_tracer for v in values):
        from ._trace import concat_tracers
        return concat_tracers(values, dim)

    def inner_concat(*values):
        broadcast_shape: Shape = values[0].shape  # merge_shapes(*[t.shape.with_sizes([None] * t.shape.rank) for t in values])
        dim_index = broadcast_shape.index(dim)
        natives = [v.native(order=broadcast_shape.names) for v in values]
        concatenated = choose_backend(*natives).concat(natives, dim_index)
        if all([v.shape.get_item_names(dim) is not None or v.shape.get_size(dim) == 0 for v in values]):
            broadcast_shape = broadcast_shape.with_dim_size(dim, sum([v.shape.get_item_names(dim) or () for v in values], ()))
        else:
            broadcast_shape = broadcast_shape.with_dim_size(dim, sum([v.shape.get_size(dim) for v in values]))
        return NativeTensor(concatenated, broadcast_shape)

    result = broadcast_op(inner_concat, values)
    return result


def pad(value: Tensor, widths: Union[dict, tuple, list], mode: Union['e_.Extrapolation', Tensor, Number, str, dict] = 0, **kwargs) -> Tensor:
    """
    Pads a tensor along the specified dimensions, determining the added values using the given extrapolation.
    Unlike `Extrapolation.pad()`, this function can handle negative widths which slice off outer values.

    Args:
        value: `Tensor` to be padded
        widths: Number of values to add at the edge of `value`. Negative values can be used to slice off edge values. Must be one of the following:

            * `tuple` containing `(lower: int, upper: int)`. This will pad all non-batch dimensions by `lower` and `upper` at the lower and upper edge, respectively.
            * `dict` mapping `dim: str -> (lower: int, upper: int)`
            * Sequence of slicing `dict`s. This will add all values specified by the slicing dicts and is the inverse operation to `slice_off`. Exactly one value in each slicing dict must be a `slice` object.

        mode: Padding mode used to determine values added from positive `widths`.
            Must be one of the following: `Extrapolation`, `Tensor` or number for constant extrapolation, name of extrapolation as `str`.
        kwargs: Additional padding arguments.
            These are ignored by the standard extrapolations defined in `phiml.math.extrapolation` but can be used to pass additional contextual information to custom extrapolations.

    Returns:
        Padded `Tensor`

    Examples:
        >>> math.pad(math.ones(spatial(x=10, y=10)), {'x': (1, 1), 'y': (2, 1)}, 0)
        (xˢ=12, yˢ=13) 0.641 ± 0.480 (0e+00...1e+00)

        >>> math.pad(math.ones(spatial(x=10, y=10)), {'x': (1, -1)}, 0)
        (xˢ=10, yˢ=10) 0.900 ± 0.300 (0e+00...1e+00)
    """
    mode = e_.as_extrapolation(mode)
    if isinstance(widths, (tuple, list)):
        if len(widths) == 0 or isinstance(widths[0], dict):  # add sliced-off slices
            return _pad_slices(value, widths, mode, **kwargs)
        if len(widths) == 2 and isinstance(widths[0], int) and isinstance(widths[1], int):  # (lower, upper)
            assert non_batch(value).rank == 1, f"Can only pad 1D tensors (excluding batch dims) when widths=(lower, upper) but got {shape(value)} and widths={widths}"
            widths = {non_batch(value).name: widths}
        else:  # ((lo0, up0), (lo1, up1), ...)
            assert len(widths) == non_batch(value), f"Cannot pad tensor with non-batch dims {non_batch(value)} by widths {widths}. Sizes must match."
            warnings.warn("Padding by sequence of (lower, upper) is not recommended. Please use a dict instead.", SyntaxWarning, stacklevel=2)
            widths = {dim: w for dim, w in zip(non_batch(value).names, widths)}
    has_negative_widths = any(w0 < 0 or w1 < 0 for w0, w1 in widths.values())
    has_positive_widths = any(w0 > 0 or w1 > 0 for w0, w1 in widths.values())
    slices = None
    if has_negative_widths:
        slices = {dim: slice(max(0, -w[0]), min(0, w[1]) or None) for dim, w in widths.items()}
        widths = {dim: (max(0, w[0]), max(0, w[1])) for dim, w in widths.items()}
    result_padded = mode.pad(value, widths, **kwargs) if has_positive_widths else value
    result_sliced = result_padded[slices] if has_negative_widths else result_padded
    return result_sliced


def _pad_slices(x: Tensor, pad_slices: Sequence[Dict[str, Any]], mode: 'e_.Extrapolation', **kwargs) -> Tensor:
    def matches_i(pad_slice: Dict[str, int], sel: Dict[str, int]):
        for dim, s in pad_slice.items():
            if dim in sel:
                if s != sel[dim]:
                    return False
        return True

    def to_lower_upper(s: slice, size: int) -> Tuple[int, int]:
        if not s.start:  # pad left
            return s.stop, 0
        else:  # pad right
            return 0, (-s.start if s.start < 0 else size - s.start)

    sel_dims = set().union(*[{dim for dim, s in s_dict.items() if not isinstance(s, slice)} for s_dict in pad_slices])
    sel_dims = x.shape.only(sel_dims)
    pad_slices = [x.shape.resolve_index(ps) for ps in pad_slices]
    padded_slices = []
    for i in sel_dims.meshgrid():
        shape_i = x.shape.after_gather(i)
        pad_slices_i = [{dim: s for dim, s in ps.items() if dim not in i} for ps in pad_slices if matches_i(ps, i)]
        widths_i = [{dim: to_lower_upper(s, shape_i.get_size(dim)) for dim, s in ps.items()} for ps in pad_slices_i]
        sum_widths_i = {dim: np.asarray([0, 0]) for dim in set().union(*widths_i)}
        for width_i in widths_i:
            dim, lu = next(iter(width_i.items()))
            sum_widths_i[dim] += lu
        padded_slices.append(mode[i].pad(x[i], {k: tuple(v) for k, v in sum_widths_i.items()}, **kwargs))
    return stack(padded_slices, sel_dims)
    # for w in widths:
    #     selection = {dim: i for dim, i in w.items() if not isinstance(i, slice)}
    #     value_sel = value[selection]
    #     slice_ = {dim: to_lower_upper(s, value_sel.shape.get_size(dim)) for dim, s in w.items() if isinstance(s, slice)}
    #     assert len(slice_) == 1, f"Each slicing dict must contain one slice() value when padding by slices but got {w}"
    #     pad_dim = next(iter(slice_))
    #     v = mode[selection].pad_values(value_sel, width, pad_dim, **kwargs)


def closest_grid_values(grid: Tensor,
                        coordinates: Tensor,
                        extrap: 'e_.Extrapolation',
                        stack_dim_prefix='closest_',
                        **kwargs):
    """
    Finds the neighboring grid points in all spatial directions and returns their values.
    The result will have 2^d values for each vector in coordiantes in d dimensions.

    Args:
      grid: grid data. The grid is spanned by the spatial dimensions of the tensor
      coordinates: tensor with 1 channel dimension holding vectors pointing to locations in grid index space
      extrap: grid extrapolation
      stack_dim_prefix: For each spatial dimension `dim`, stacks lower and upper closest values along dimension `stack_dim_prefix+dim`.
      kwargs: Additional information for the extrapolation.

    Returns:
      Tensor of shape (batch, coord_spatial, grid_spatial=(2, 2,...), grid_channel)

    """
    return broadcast_op(functools.partial(_closest_grid_values, extrap=extrap, stack_dim_prefix=stack_dim_prefix, pad_kwargs=kwargs), [grid, coordinates])


def _closest_grid_values(grid: Tensor,
                         coordinates: Tensor,
                         extrap: 'e_.Extrapolation',
                         stack_dim_prefix: str,
                         pad_kwargs: dict):
    # alternative method: pad array for all 2^d combinations, then stack to simplify gather.
    # --- Pad tensor where transform is not possible ---
    non_copy_pad = {dim: (0 if extrap.is_copy_pad(dim, False) else 1, 0 if extrap.is_copy_pad(dim, True) else 1) for dim in grid.shape.spatial.names}
    grid = extrap.pad(grid, non_copy_pad, **pad_kwargs)
    coordinates += wrap([not extrap.is_copy_pad(dim, False) for dim in grid.shape.spatial.names], channel('vector'))
    # --- Transform coordiantes ---
    min_coords = to_int32(floor(coordinates))
    max_coords = extrap.transform_coordinates(min_coords + 1, grid.shape)
    min_coords = extrap.transform_coordinates(min_coords, grid.shape)

    def left_right(is_hi_by_axis_left, ax_idx):
        is_hi_by_axis_right = is_hi_by_axis_left | np.array([ax == ax_idx for ax in range(grid.shape.spatial_rank)])
        coords_left = where(is_hi_by_axis_left, max_coords, min_coords)
        coords_right = where(is_hi_by_axis_right, max_coords, min_coords)
        if ax_idx == grid.shape.spatial_rank - 1:
            values_left = gather(grid, coords_left)
            values_right = gather(grid, coords_right)
        else:
            values_left = left_right(is_hi_by_axis_left, ax_idx + 1)
            values_right = left_right(is_hi_by_axis_right, ax_idx + 1)
        return stack_tensors([values_left, values_right], channel(f"{stack_dim_prefix}{grid.shape.spatial.names[ax_idx]}"))

    result = left_right(np.array([False] * grid.shape.spatial_rank), 0)
    return result


def grid_sample(grid: Tensor, coordinates: Tensor, extrap: Union['e_.Extrapolation', float, str], **kwargs):
    """
    Samples values of `grid` at the locations referenced by `coordinates`.
    Values lying in between sample points are determined via linear interpolation.

    For values outside the valid bounds of `grid` (`coord < 0 or coord > grid.shape - 1`), `extrap` is used to determine the neighboring grid values.
    If the extrapolation does not support resampling, the grid is padded by one cell layer before resampling.
    In that case, values lying further outside will not be sampled according to the extrapolation.

    Args:
        grid: Grid with at least one spatial dimension and no instance dimensions.
        coordinates: Coordinates with a single channel dimension called `'vector'`.
            The size of the `vector` dimension must match the number of spatial dimensions of `grid`.
        extrap: Extrapolation used to determine the values of `grid` outside its valid bounds.
        kwargs: Additional information for the extrapolation.

    Returns:
        `Tensor` with channel dimensions of `grid`, spatial and instance dimensions of `coordinates` and combined batch dimensions.
    """
    extrap = e_.as_extrapolation(extrap) if extrap is not None else None
    result = broadcast_op(functools.partial(_grid_sample, extrap=extrap, pad_kwargs=kwargs), [grid, coordinates])
    return result


def _grid_sample(grid: Tensor, coordinates: Tensor, extrap: Union['e_.Extrapolation', None], pad_kwargs: dict):
    if grid.shape.batch == coordinates.shape.batch or grid.shape.batch.volume == 1 or coordinates.shape.batch.volume == 1:
        # call backend.grid_sample()
        batch = grid.shape.batch & coordinates.shape.batch
        backend = choose_backend_t(grid, coordinates)
        result = NotImplemented
        if extrap is None:
            result = backend.grid_sample(reshaped_native(grid, [batch, *grid.shape.spatial, grid.shape.channel]),
                                         reshaped_native(coordinates, [batch, *coordinates.shape.instance, *coordinates.shape.spatial, 'vector']),
                                         'undefined')
        elif extrap.native_grid_sample_mode:
            result = backend.grid_sample(reshaped_native(grid, [batch, *grid.shape.spatial, grid.shape.channel]),
                                         reshaped_native(coordinates, [batch, *coordinates.shape.instance, *coordinates.shape.spatial, 'vector']),
                                         extrap.native_grid_sample_mode)
        if result is NotImplemented:
            # pad one layer
            grid_padded = pad(grid, {dim: (1, 1) for dim in grid.shape.spatial.names}, extrap or e_.ZERO, **pad_kwargs)
            if extrap is not None:
                from .extrapolation import _CopyExtrapolation
                if isinstance(extrap, _CopyExtrapolation):
                    inner_coordinates = extrap.transform_coordinates(coordinates, grid.shape) + 1
                else:
                    inner_coordinates = extrap.transform_coordinates(coordinates + 1, grid_padded.shape)
            else:
                inner_coordinates = coordinates + 1
            result = backend.grid_sample(reshaped_native(grid_padded, [batch, *grid_padded.shape.spatial.names, grid.shape.channel]),
                                         reshaped_native(inner_coordinates, [batch, *coordinates.shape.instance, *coordinates.shape.spatial, 'vector']),
                                         'boundary')
        if result is not NotImplemented:
            result = reshaped_tensor(result, [grid.shape.batch & coordinates.shape.batch, *coordinates.shape.instance, *coordinates.shape.spatial, grid.shape.channel])
            return result
    # fallback to slower grid sampling
    neighbors = _closest_grid_values(grid, coordinates, extrap or e_.ZERO, '_closest_', pad_kwargs)
    binary = meshgrid(channel, **{f'_closest_{dim}': (0, 1) for dim in grid.shape.spatial.names}, stack_dim=channel(coordinates))
    right_weights = coordinates % 1
    weights = prod(binary * right_weights + (1 - binary) * (1 - right_weights), 'vector')
    result = sum_(neighbors * weights, dim=[f"_closest_{dim}" for dim in grid.shape.spatial.names])
    return result


def broadcast_dims(*tensors: Tensor):
    iter_dims = set()
    for tensor in tensors:
        iter_dims.update(shape(tensor).shape.without('dims').names)
        if isinstance(tensor, TensorStack) and tensor.requires_broadcast:
            iter_dims.add(tensor._stack_dim.name)
        # --- remove iter_dims for which the sizes vary among tensors ---
        for dim in tuple(iter_dims):
            sizes = [t.shape.get_size(dim) for t in tensors if dim in t.shape]
            if not all(s == sizes[0] for s in sizes[1:]):
                iter_dims.remove(dim)
    return iter_dims


def broadcast_op(operation: Callable,
                 tensors: Union[tuple, list],
                 iter_dims: Union[set, tuple, list, Shape] = None,
                 no_return=False):
    iter_dims = broadcast_dims(*tensors) if iter_dims is None else iter_dims
    if len(iter_dims) == 0:
        return operation(*tensors)
    else:
        if isinstance(iter_dims, Shape):
            iter_dims = iter_dims.names
        dim = next(iter(iter_dims))
        dim_type = None
        size = None
        item_names = None
        unstacked = []
        for tensor in tensors:
            if dim in tensor.shape.names:
                unstacked_tensor = tensor._unstack(dim)
                unstacked.append(unstacked_tensor)
                if size is None:
                    size = len(unstacked_tensor)
                    dim_type = tensor.shape.get_type(dim)
                else:
                    assert size == len(unstacked_tensor)
                    assert dim_type == tensor.shape.get_type(dim)
                if item_names is None:
                    item_names = tensor.shape.get_item_names(dim)
            else:
                unstacked.append(tensor)
        result_unstacked = []
        for i in range(size):
            gathered = [t[i] if isinstance(t, tuple) else t for t in unstacked]
            result_unstacked.append(broadcast_op(operation, gathered, iter_dims=set(iter_dims) - {dim}))
        if not no_return:
            return TensorStack(result_unstacked, Shape((size,), (dim,), (dim_type,), (item_names,)))


def where(condition: Union[Tensor, float, int],
          value_true: Union[Tensor, float, int, Any] = None,
          value_false: Union[Tensor, float, int, Any] = None):
    """
    Builds a tensor by choosing either values from `value_true` or `value_false` depending on `condition`.
    If `condition` is not of type boolean, non-zero values are interpreted as True.
    
    This function requires non-None values for `value_true` and `value_false`.
    To get the indices of True / non-zero values, use :func:`nonzero`.

    Args:
      condition: determines where to choose values from value_true or from value_false
      value_true: Values to pick where `condition != 0 / True`
      value_false: Values to pick where `condition == 0 / False`

    Returns:
        `Tensor` containing dimensions of all inputs.
    """
    if value_true is None:
        assert value_false is None, f"where can be used either with value_true and value_false or without both but got only value_false"
        warnings.warn("Use nonzero() instead of where() to get indices of non-zero elements.", SyntaxWarning, stacklevel=2)
        return nonzero(condition)
    from .extrapolation import Extrapolation, where as ext_where
    if isinstance(value_true, Extrapolation) or isinstance(value_false, Extrapolation):
        return ext_where(condition, value_true, value_false)
    condition = wrap(condition)
    value_true = wrap(value_true)
    value_false = wrap(value_false)

    def inner_where(c: Tensor, vt: Tensor, vf: Tensor):
        if vt._is_tracer or vf._is_tracer or c._is_tracer:
            return c * vt + (1 - c) * vf  # ToDo this does not take NaN into account
        if is_sparse(vt) or is_sparse(vf):
            if same_sparsity_pattern(vt, vf, allow_const=True) and same_sparsity_pattern(c, vt, allow_const=True):
                c_values = c._values if is_sparse(c) else c
                vt_values = vt._values if is_sparse(vt) else vt
                vf_values = vf._values if is_sparse(vf) else vf
                result_values = where(c_values, vt_values, vf_values)
                return c._with_values(result_values)
            raise NotImplementedError
        shape, (c, vt, vf) = broadcastable_native_tensors(c, vt, vf)
        result = choose_backend(c, vt, vf).where(c, vt, vf)
        return NativeTensor(result, shape)

    return broadcast_op(inner_where, [condition, value_true, value_false])


def nonzero(value: Tensor, list_dim: Union[Shape, str] = instance('nonzero'), index_dim: Shape = channel('vector')):
    """
    Get spatial indices of non-zero / True values.
    
    Batch dimensions are preserved by this operation.
    If channel dimensions are present, this method returns the indices where any component is nonzero.

    Implementations:

    * NumPy: [`numpy.argwhere`](https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html)
    * PyTorch: [`torch.nonzero`](https://pytorch.org/docs/stable/generated/torch.nonzero.html)
    * TensorFlow: [`tf.where(tf.not_equal(values, 0))`](https://www.tensorflow.org/api_docs/python/tf/where)
    * Jax: [`jax.numpy.nonzero`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.nonzero.html)

    Args:
        value: spatial tensor to find non-zero / True values in.
        list_dim: Dimension listing non-zero values.
        index_dim: Index dimension.

    Returns:
        `Tensor` of shape (batch dims..., `list_dim`=#non-zero, `index_dim`=value.shape.spatial_rank)

    """
    if value.shape.channel_rank > 0:
        value = sum_(abs(value), value.shape.channel)
    if isinstance(list_dim, str):
        list_dim = instance(list_dim)
    def unbatched_nonzero(value: Tensor):
        if isinstance(value, CompressedSparseMatrix):
            value = value.decompress()
        if isinstance(value, SparseCoordinateTensor):
            nonzero_values = nonzero(value._values)
            nonzero_indices = value._indices[nonzero_values]
            index_dim_ = index_dim.with_size(channel(value._indices).item_names[0])
            return rename_dims(rename_dims(nonzero_indices, instance, list_dim), channel, index_dim_)
        else:
            dims = value.shape.non_channel
            native = reshaped_native(value, [*dims])
            backend = choose_backend(native)
            indices = backend.nonzero(native)
            indices_shape = Shape(backend.staticshape(indices), (list_dim.name, index_dim.name), (list_dim.type, index_dim.type), (None, dims.names))
            return NativeTensor(indices, indices_shape)
    return broadcast_op(unbatched_nonzero, [value], iter_dims=value.shape.batch.names)


def nonzero_slices(x: Tensor):
    """

    Args:
        x: Concrete tensor.

    Returns:

    """
    assert x.available, f"nonzero_slices requires a concrete tensor but got {x}"
    assert x.rank == 1, f"nonzero_slices requires a 1D tensor but got {x.shape}"
    if x.shape.volume == 0:
        return []
    dim = x.shape.name
    if x.dtype.kind != bool:
        x = x != 0
    slices = []
    start = None
    for i, val in enumerate(x):
        if val:
            if start is None:
                start = i
        else:
            if start is not None:
                slices.append({dim: slice(start, i)})
                start = None
    if start is not None:
        slices.append({dim: slice(start, None)})
    return slices


def reduce_(f, value, dims, require_all_dims_present=False, required_kind: type = None):
    if not dims:
        return value
    else:
        if isinstance(value, (tuple, list)):
            values = [wrap(v) for v in value]
            value = stack_tensors(values, instance('0'))
            dims = value.shape.only(dims)
            assert '0' in dims, "When passing a sequence of tensors to be reduced, the sequence dimension '0' must be reduced."
        elif isinstance(value, Layout):
            if value.shape.without(value._stack_dim).only(dims):  # reduce some inner
                def inner_reduce(v):
                    if required_kind is not None:
                        if isinstance(v, Tensor):
                            v = cast(v, required_kind)
                        else:
                            v = required_kind(v)
                    return f(wrap(v), shape(v).only(dims))

                value = tree_map(inner_reduce, value)
            if value._stack_dim.without(dims).is_empty:  # reduce all outer
                values = value._as_list()
                dims = batch('_flat_layout')
                value = wrap(values, dims)
        else:
            value = wrap(value)
        dims = value.shape.only(dims)
        if require_all_dims_present and any(d not in value.shape for d in dims):
            raise ValueError(f"Cannot sum dimensions {dims} because tensor {value.shape} is missing at least one of them")
        return f(value._simplify(), dims)


def sum_(value: Union[Tensor, list, tuple, Number, bool], dim: DimFilter = non_batch) -> Tensor:
    """
    Sums `values` along the specified dimensions.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return reduce_(_sum, bool_to_int(value), dim, require_all_dims_present=True)


def _sum(value: Tensor, dims: Shape) -> Tensor:
    if not dims:
        return value
    if isinstance(value, NativeTensor):
        result = value.default_backend.sum(value._native, value._native_shape.indices(dims)) * value.collapsed_dims.only(dims).volume
        return NativeTensor(result, value._native_shape.without(dims), value.shape.without(dims))
    elif isinstance(value, TensorStack):
        reduced_inners = [_sum(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: x + y, reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    elif isinstance(value, (CompressedSparseMatrix, SparseCoordinateTensor)):
        return sparse_sum(value, dims)
    else:
        raise ValueError(type(value))


def prod(value: Union[Tensor, list, tuple, Number, bool], dim: DimFilter = non_batch) -> Tensor:
    """
    Multiplies `values` along the specified dimensions.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return reduce_(_prod, value, dim, require_all_dims_present=True)


def _prod(value: Tensor, dims: Shape) -> Tensor:
    if isinstance(value, NativeTensor):
        result = value.default_backend.prod(value._native, value._native_shape.indices(dims)) ** value.collapsed_dims.only(dims).volume
        return NativeTensor(result, value._native_shape.without(dims), value.shape.without(dims))
    elif isinstance(value, TensorStack):
        reduced_inners = [_prod(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: x * y, reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    else:
        raise ValueError(type(value))


def mean(value: Union[Tensor, list, tuple, Number, bool], dim: DimFilter = non_batch) -> Tensor:
    """
    Computes the mean over `values` along the specified dimensions.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return reduce_(_mean, value, dim)


def _mean(value: Tensor, dims: Shape) -> Tensor:
    if not dims:
        return value
    if isinstance(value, NativeTensor):
        result = value.default_backend.mean(value._native, value._native_shape.indices(dims))
        return NativeTensor(result, value._native_shape.without(dims), value.shape.without(dims))
    elif isinstance(value, TensorStack):
        reduced_inners = [_mean(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: x + y, reduced_inners) / len(reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    else:
        raise ValueError(type(value))


def std(value: Union[Tensor, list, tuple, Number, bool], dim: DimFilter = non_batch) -> Tensor:
    """
    Computes the standard deviation over `values` along the specified dimensions.

    *Warning*: The standard deviation of non-uniform tensors along the stack dimension is undefined.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    if not dim:
        warnings.warn("std along empty shape returns 0", RuntimeWarning, stacklevel=2)
        return zeros_like(value)
    if not callable(dim) and set(parse_dim_order(dim)) - set(value.shape.names):
        return zeros_like(value)  # std along constant dim is 0
    return reduce_(_std, value, dim)


def _std(value: Tensor, dims: Shape) -> Tensor:
    if value.shape.is_uniform:
        result = value.default_backend.std(value.native(value.shape), value.shape.indices(dims))
        return NativeTensor(result, value.shape.without(dims))
    else:
        non_uniform_dim = value.shape.shape.without('dims')
        assert non_uniform_dim.only(dims).is_empty, f"Cannot compute std along non-uniform dims {dims}. shape={value.shape}"
        return stack([_std(t, dims) for t in value._unstack(non_uniform_dim.name)], non_uniform_dim)


def any_(boolean_tensor: Union[Tensor, list, tuple], dim: DimFilter = non_batch) -> Tensor:
    """
    Tests whether any entry of `boolean_tensor` is `True` along the specified dimensions.

    Args:
        boolean_tensor: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return reduce_(_any, boolean_tensor, dim, required_kind=bool)


def _any(value: Tensor, dims: Shape) -> Tensor:
    if isinstance(value, NativeTensor):
        result = value.default_backend.any(value._native, value._native_shape.indices(dims))
        return NativeTensor(result, value._native_shape.without(dims), value.shape.without(dims))
    elif isinstance(value, TensorStack):
        reduced_inners = [_any(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: x | y, reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    elif isinstance(value, (CompressedSparseMatrix, SparseCoordinateTensor)):
        return sparse_sum(to_int32(value), dims) > 0
    else:
        raise ValueError(type(value))


def all_(boolean_tensor: Union[Tensor, list, tuple, Number, bool], dim: DimFilter = non_batch) -> Tensor:
    """
    Tests whether all entries of `boolean_tensor` are `True` along the specified dimensions.

    Args:
        boolean_tensor: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return reduce_(_all, boolean_tensor, dim, required_kind=bool)


def _all(value: Tensor, dims: Shape) -> Tensor:
    if isinstance(value, NativeTensor):
        result = value.default_backend.all(value.native(value.shape), value.shape.indices(dims))
        return NativeTensor(result, value.shape.without(dims))
    elif isinstance(value, TensorStack):
        reduced_inners = [_all(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: x & y, reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    elif isinstance(value, (SparseCoordinateTensor, CompressedSparseMatrix)):
        if sparse_dims(value) in dims:
            values_all = _all(value._values, dims.without(sparse_dims(value)) & instance(value._values))
            return all_([values_all, value._default], '0') if value._default is not None else values_all
    raise ValueError(type(value))


def max_(value: Union[Tensor, list, tuple, Number, bool], dim: DimFilter = non_batch) -> Tensor:
    """
    Determines the maximum value of `values` along the specified dimensions.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return reduce_(_max, value, dim)


def _max(value: Tensor, dims: Shape) -> Tensor:
    if isinstance(value, NativeTensor):
        result = value.default_backend.max(value.native(value.shape), value.shape.indices(dims))
        return NativeTensor(result, value.shape.without(dims))
    elif isinstance(value, TensorStack):
        reduced_inners = [_max(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: maximum(x, y), reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    elif isinstance(value, (SparseCoordinateTensor, CompressedSparseMatrix)):
        return sparse_max(value, dims)
    raise ValueError(type(value))


def min_(value: Union[Tensor, list, tuple, Number, bool], dim: DimFilter = non_batch) -> Tensor:
    """
    Determines the minimum value of `values` along the specified dimensions.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor` without the reduced dimensions.
    """
    return reduce_(_min, value, dim)


def _min(value: Tensor, dims: Shape) -> Tensor:
    if isinstance(value, NativeTensor):
        result = value.default_backend.min(value.native(value.shape), value.shape.indices(dims))
        return NativeTensor(result, value.shape.without(dims))
    elif isinstance(value, TensorStack):
        reduced_inners = [_min(t, dims.without(value._stack_dim)) for t in value._tensors]
        return functools.reduce(lambda x, y: minimum(x, y), reduced_inners) if value._stack_dim in dims else TensorStack(reduced_inners, value._stack_dim)
    elif isinstance(value, (SparseCoordinateTensor, CompressedSparseMatrix)):
        return sparse_min(value, dims)
    raise ValueError(type(value))


def finite_min(value, dim: DimFilter = non_batch, default: Union[complex, float] = float('NaN')):
    """
    Finds the minimum along `dim` ignoring all non-finite values.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

        default: Value to use where no finite value was encountered.

    Returns:
        `Tensor` without the reduced dimensions.
    """
    value_inf = where(is_finite(value), value, float('inf'))
    result_inf = min_(value_inf, dim)
    return where(is_finite(result_inf), result_inf, default)


def finite_max(value, dim: DimFilter = non_batch, default: Union[complex, float] = float('NaN')):
    """
    Finds the maximum along `dim` ignoring all non-finite values.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

        default: Value to use where no finite value was encountered.

    Returns:
        `Tensor` without the reduced dimensions.
    """
    value_inf = where(is_finite(value), value, float('-inf'))
    result_inf = max_(value_inf, dim)
    return where(is_finite(result_inf), result_inf, default)


def finite_sum(value, dim: DimFilter = non_batch, default: Union[complex, float] = float('NaN')):
    """
    Sums all finite values in `value` along `dim`.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

        default: Value to use where no finite value was encountered.

    Returns:
        `Tensor` without the reduced dimensions.
    """
    finite = is_finite(value)
    summed = sum_(where(finite, value, 0), dim)
    return where(any_(finite, dim), summed, default)


def finite_mean(value, dim: DimFilter = non_batch, default: Union[complex, float] = float('NaN')):
    """
    Computes the mean value of all finite values in `value` along `dim`.

    Args:
        value: `Tensor` or `list` / `tuple` of Tensors.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

        default: Value to use where no finite value was encountered.

    Returns:
        `Tensor` without the reduced dimensions.
    """
    finite = is_finite(value)
    summed = sum_(where(finite, value, 0), dim)
    count = sum_(finite, dim)
    mean_nan = summed / count
    return where(is_finite(mean_nan), mean_nan, default)


def lookup_where(native_index_fn: Callable, value: Union[Tensor, PhiTreeNode], key: Tensor, dim: DimFilter):
    dims = key.shape.only(dim)
    keep = key.shape.without(dims)
    assert dim, f"No dimensions {dim} present on key {key.shape}"
    v_native = reshaped_native(key, [keep, dims])
    idx_native = native_index_fn(v_native)
    multi_idx_native = choose_backend(idx_native).unravel_index(idx_native[:, 0], dims.sizes)
    idx = reshaped_tensor(multi_idx_native, [keep.as_batch(), channel(_index=dims)])
    def lookup(t: Tensor):
        keep_t = t.shape.without(dims)
        sel = rename_dims(t, keep_t, batch)[idx]
        return rename_dims(rename_dims(sel, keep_t.names, keep_t), keep.names, keep)
    result = tree_map(lookup, value)
    return result


def at_max(value, key: Tensor, dim: DimFilter = non_batch):
    """
    Looks up the values of `value` at the positions where the maximum values in `key` are located along `dim`.

    See Also:
        `at_min`, `phiml.math.max`.

    Args:
        value: Tensors or trees from which to lookup and return values. These tensors are indexed at the maximum index in `key´.
        key: `Tensor` containing at least one dimension of `dim`. The maximum index of `key` is determined.
        dim: Dimensions along which to compute the maximum of `key`.

    Returns:
        The values of `other_tensors` at the positions where the maximum values in `value` are located along `dim`.
    """
    return lookup_where(lambda v: choose_backend(v).argmax(v, 1, keepdims=True), value, key, dim)


def at_min(value, key: Tensor, dim: DimFilter = non_batch):
    """
    Looks up the values of `value` at the positions where the minimum values in `key` are located along `dim`.

    See Also:
        `at_max`, `phiml.math.min`.

    Args:
        value: Tensors or trees from which to lookup and return values. These tensors are indexed at the minimum index in `key´.
        key: `Tensor` containing at least one dimension of `dim`. The minimum index of `key` is determined.
        dim: Dimensions along which to compute the minimum of `key`.

    Returns:
        The values of `other_tensors` at the positions where the minimum values in `value` are located along `dim`.
    """
    return lookup_where(lambda v: choose_backend(v).argmin(v, 1, keepdims=True), value, key, dim)


def argmax(x: Tensor, dim: DimFilter, index_dim=channel('index')):
    """
    Finds the maximum value along one or multiple dimensions and returns the corresponding index.

    See Also:
        `argmin`, `at_max`.

    Args:
        x: `Tensor`
        dim: Dimensions along which the maximum should be determined. These are reduced in the operation.
        index_dim: Dimension listing the index components for multidimensional argmax.

    Returns:
        Index tensor `idx`, such that `x[idx] = max(x)`.
    """
    dims = x.shape.only(dim)
    keep = x.shape.without(dims)
    assert dim, f"No dimensions {dim} present on key {x.shape}"
    if isinstance(x, (SparseCoordinateTensor, CompressedSparseMatrix)):
        if dims in sparse_dims(x):
            max_val = max_(x, dim)
            is_max = x == max_val
            is_max_idx = nonzero(is_max, list_dim=instance('true_values'))
            scatter_val = is_max_idx[dims.only(sparse_dims(x)).name_list]
            scatter_idx = is_max_idx[sparse_dims(x).without(dims).name_list]
            result_shape = max_val.shape & channel(scatter_val)
            result = scatter(result_shape, scatter_idx, scatter_val, mode='update', default=-1)
            return rename_dims(result, channel(scatter_val), index_dim)
        else:
            raise NotImplementedError
    v_native = reshaped_native(x, [keep, dims])
    idx_native = x.default_backend.argmax(v_native, 1, keepdims=True)
    multi_idx_native = choose_backend(idx_native).unravel_index(idx_native[:, 0], dims.sizes)
    return reshaped_tensor(multi_idx_native, [keep, index_dim.with_size(dims)])


def argmin(x: Tensor, dim: DimFilter, index_dim=channel('index')):
    """
    Finds the minimum value along one or multiple dimensions and returns the corresponding index.

    See Also:
        `argmax`, `at_min`.

    Args:
        x: `Tensor`
        dim: Dimensions along which the minimum should be determined. These are reduced in the operation.
        index_dim: Dimension listing the index components for multidimensional argmin.

    Returns:
        Index tensor `idx`, such that `x[idx] = min(x)`.
    """
    dims = x.shape.only(dim)
    keep = x.shape.without(dims)
    assert dim, f"No dimensions {dim} present on key {x.shape}"
    v_native = reshaped_native(x, [keep, dims])
    idx_native = x.default_backend.argmin(v_native, 1, keepdims=True)
    multi_idx_native = choose_backend(idx_native).unravel_index(idx_native[:, 0], dims.sizes)
    return reshaped_tensor(multi_idx_native, [keep, index_dim.with_size(dims)])


def quantile(value: Tensor,
             quantiles: Union[float, tuple, list, Tensor],
             dim: DimFilter = non_batch):
    """
    Compute the q-th quantile of `value` along `dim` for each q in `quantiles`.

    Implementations:

    * NumPy: [`quantile`](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)
    * PyTorch: [`quantile`](https://pytorch.org/docs/stable/generated/torch.quantile.html#torch.quantile)
    * TensorFlow: [`tfp.stats.percentile`](https://www.tensorflow.org/probability/api_docs/python/tfp/stats/percentile)
    * Jax: [`quantile`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.quantile.html)

    Args:
        value: `Tensor`
        quantiles: Single quantile or tensor of quantiles to compute.
            Must be of type `float`, `tuple`, `list` or `Tensor`.
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to reduce the sequence of Tensors

    Returns:
        `Tensor` with dimensions of `quantiles` and non-reduced dimensions of `value`.
    """
    dims = value.shape.only(dim)
    native_values = reshaped_native(value, [*value.shape.without(dims), value.shape.only(dims)])
    backend = choose_backend(native_values)
    q = wrap(quantiles, default_list_dim=instance('quantiles'))
    native_quantiles = reshaped_native(q, [q.shape])
    native_result = backend.quantile(native_values, native_quantiles)
    return reshaped_tensor(native_result, [q.shape, *value.shape.without(dims)])


def median(value, dim: DimFilter = non_batch):
    """
    Reduces `dim` of `value` by picking the median value.
    For odd dimension sizes (ambigous choice), the linear average of the two median values is computed.

    Currently implemented via `quantile()`.

    Args:
        value: `Tensor`
        dim: Dimension or dimensions to be reduced. One of

            * `None` to reduce all non-batch dimensions
            * `str` containing single dimension or comma-separated list of dimensions
            * `Tuple[str]` or `List[str]`
            * `Shape`
            * `batch`, `instance`, `spatial`, `channel` to select dimensions by type
            * `'0'` when `isinstance(value, (tuple, list))` to add up the sequence of Tensors

    Returns:
        `Tensor`
    """
    return reduce_(_median, value, dim)


def _median(value: Tensor, dims: Shape) -> Tensor:
    if not dims:
        return value
    if isinstance(value, NativeTensor):
        return quantile(value, 0.5, dims)
    elif isinstance(value, TensorStack):
        reduced_inners = [_median(t, dims.without(value._stack_dim)) for t in value._tensors]
        if value._stack_dim in dims:
            raise NotImplementedError  # return median(reduced_inners)
        else:
            return TensorStack(reduced_inners, value._stack_dim)
    else:
        raise ValueError(type(value))


def dot(x: Tensor,
        x_dims: DimFilter,
        y: Tensor,
        y_dims: DimFilter) -> Tensor:
    """
    Computes the dot product along the specified dimensions.
    Contracts `x_dims` with `y_dims` by first multiplying the elements and then summing them up.

    For one dimension, this is equal to matrix-matrix or matrix-vector multiplication.

    The function replaces the traditional `dot` / `tensordot` / `matmul` / `einsum` functions.

    * NumPy: [`numpy.tensordot`](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html), [`numpy.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
    * PyTorch: [`torch.tensordot`](https://pytorch.org/docs/stable/generated/torch.tensordot.html#torch.tensordot), [`torch.einsum`](https://pytorch.org/docs/stable/generated/torch.einsum.html)
    * TensorFlow: [`tf.tensordot`](https://www.tensorflow.org/api_docs/python/tf/tensordot), [`tf.einsum`](https://www.tensorflow.org/api_docs/python/tf/einsum)
    * Jax: [`jax.numpy.tensordot`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tensordot.html), [`jax.numpy.einsum`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.einsum.html)

    Args:
        x: First `Tensor`
        x_dims: Dimensions of `x` to reduce against `y`
        y: Second `Tensor`
        y_dims: Dimensions of `y` to reduce against `x`.

    Returns:
        Dot product as `Tensor`.
    """
    x_dims = x.shape.only(x_dims)
    y_dims = y.shape.only(y_dims)
    if not x_dims:
        return x * sum_(y, y_dims)
    if not y_dims:
        return sum_(x, x_dims) * y

    def tensor_dot(x, y):
        if is_sparse(x) or is_sparse(y):
            if x_dims.isdisjoint(sparse_dims(x)) and y_dims.isdisjoint(sparse_dims(y)):
                if is_sparse(x):
                    return x._op2(y, lambda vx, vy: dot(vx, x_dims, vy, y_dims), None, 'dot', '@')
                else:
                    return y._op2(x, lambda vy, vx: dot(vx, x_dims, vy, y_dims), None, 'dot', '@')
            else:
                return sparse_dot(x, x_dims, y, y_dims)
        if x._is_tracer:
            return x._matmul(x_dims, y, y_dims)
        if y._is_tracer:
            return y._matmul(y_dims, x, x_dims)
        x_native = x.native(x.shape)
        y_native = y.native(y.shape)
        backend = choose_backend(x_native, y_native)
        remaining_shape_x = x.shape.without(x_dims)
        remaining_shape_y = y.shape.without(y_dims)
        assert x_dims.volume == y_dims.volume, f"Failed to reduce {x_dims} against {y_dims} in dot product of {x.shape} and {y.shape}. Sizes do not match."
        if remaining_shape_y.isdisjoint(remaining_shape_x):  # no shared batch dimensions -> tensordot
            result_native = backend.tensordot(x_native, x.shape.indices(x_dims), y_native, y.shape.indices(y_dims))
            result_shape = concat_shapes(remaining_shape_x, remaining_shape_y)
        else:  # shared batch dimensions -> einsum
            result_shape = merge_shapes(x.shape.without(x_dims), y.shape.without(y_dims))
            REDUCE_LETTERS = list('ijklmn')
            KEEP_LETTERS = list('abcdefgh')
            x_letters = [(REDUCE_LETTERS if dim in x_dims else KEEP_LETTERS).pop(0) for dim in x.shape.names]
            letter_map = {dim: letter for dim, letter in zip(x.shape.names, x_letters)}
            REDUCE_LETTERS = list('ijklmn')
            y_letters = []
            for dim in y.shape.names:
                if dim in y_dims:
                    y_letters.append(REDUCE_LETTERS.pop(0))
                else:
                    if dim in x.shape and dim not in x_dims:
                        y_letters.append(letter_map[dim])
                    else:
                        next_letter = KEEP_LETTERS.pop(0)
                        letter_map[dim] = next_letter
                        y_letters.append(next_letter)
            keep_letters = [letter_map[dim] for dim in result_shape.names]
            subscripts = f'{"".join(x_letters)},{"".join(y_letters)}->{"".join(keep_letters)}'
            result_native = backend.einsum(subscripts, x_native, y_native)
        return NativeTensor(result_native, result_shape)

    return broadcast_op(tensor_dot, [x, y])


def _backend_op1(x, unbound_method) -> Union[Tensor, PhiTreeNode]:
    if isinstance(x, Tensor):
        def apply_op(native_tensor):
            backend = choose_backend(native_tensor)
            return getattr(backend, unbound_method.__name__)(backend.auto_cast(native_tensor)[0])
        apply_op.__name__ = unbound_method.__name__
        return x._op1(apply_op)
    elif isinstance(x, PhiTreeNode):
        return copy_with(x, **{a: _backend_op1(getattr(x, a), unbound_method) for a in value_attributes(x)})
    else:
        backend = choose_backend(x)
        y = getattr(backend, unbound_method.__name__)(backend.auto_cast(x)[0])
        return y


def abs_(x: TensorOrTree) -> TensorOrTree:
    """
    Computes *||x||<sub>1</sub>*.
    Complex `x` result in matching precision float values.

    *Note*: The gradient of this operation is undefined for *x=0*.
    TensorFlow and PyTorch return 0 while Jax returns 1.

    Args:
        x: `Tensor` or `phiml.math.magic.PhiTreeNode`

    Returns:
        Absolute value of `x` of same type as `x`.
    """
    return _backend_op1(x, Backend.abs)


def sign(x: TensorOrTree) -> TensorOrTree:
    """
    The sign of positive numbers is 1 and -1 for negative numbers.
    The sign of 0 is undefined.

    Args:
        x: `Tensor` or `phiml.math.magic.PhiTreeNode`

    Returns:
        `Tensor` or `phiml.math.magic.PhiTreeNode` matching `x`.
    """
    return _backend_op1(x, Backend.sign)


def round_(x: TensorOrTree) -> TensorOrTree:
    """ Rounds the `Tensor` or `phiml.math.magic.PhiTreeNode` `x` to the closest integer. """
    return _backend_op1(x, Backend.round)


def ceil(x: TensorOrTree) -> TensorOrTree:
    """ Computes *⌈x⌉* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.ceil)


def floor(x: TensorOrTree) -> TensorOrTree:
    """ Computes *⌊x⌋* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.floor)


def sqrt(x: TensorOrTree) -> TensorOrTree:
    """ Computes *sqrt(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.sqrt)


def exp(x: TensorOrTree) -> TensorOrTree:
    """ Computes *exp(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.exp)


def erf(x: TensorOrTree) -> TensorOrTree:
    """ Computes the error function *erf(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.erf)


def soft_plus(x: TensorOrTree) -> TensorOrTree:
    """ Computes *softplus(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.softplus)


def factorial(x: TensorOrTree) -> TensorOrTree:
    """
    Computes *factorial(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`.
    For floating-point numbers computes the continuous factorial using the gamma function.
    For integer numbers computes the exact factorial and returns the same integer type.
    However, this results in integer overflow for inputs larger than 12 (int32) or 19 (int64).
    """
    return _backend_op1(x, Backend.factorial)


def log_gamma(x: TensorOrTree) -> TensorOrTree:
    """ Computes *log(gamma(x))* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.log_gamma)


def incomplete_gamma(a: TensorOrTree, x: TensorOrTree, upper=False, regularized=True) -> TensorOrTree:
    """
    Computes the incomplete gamma function.

    Args:
        a: Positive parameter, `Tensor` or tree.
        x: Non-negative argument, `Tensor` or tree.
        upper: Whether to complete the upper integral (x to infinity) or the lower integral (0 to x).
        regularized: Whether the integral is divided by Γ(a).
    """
    call = lambda a, x: incomplete_gamma(a, x, upper=upper, regularized=regularized)
    if upper:
        reg = custom_op2(a, x, call, lambda a, x: choose_backend(a, x).gamma_inc_u(a, x), 'gamma_inc_u')
    else:
        reg = custom_op2(a, x, call, lambda a, x: choose_backend(a, x).gamma_inc_l(a, x), 'gamma_inc_l')
    return reg if regularized else reg * exp(log_gamma(a))


def to_float(x: TensorOrTree) -> TensorOrTree:
    """
    Converts the given tensor to floating point format with the currently specified precision.
    
    The precision can be set globally using `math.set_global_precision()` and locally using `with math.precision()`.
    
    See the documentation at https://tum-pbs.github.io/PhiML/Data_Types.html

    See Also:
        `cast()`.

    Args:
        x: `Tensor` or `phiml.math.magic.PhiTreeNode` to convert

    Returns:
        `Tensor` or `phiml.math.magic.PhiTreeNode` matching `x`.
    """
    return _backend_op1(x, Backend.to_float)


def to_int32(x: TensorOrTree) -> TensorOrTree:
    """ Converts the `Tensor` or `phiml.math.magic.PhiTreeNode` `x` to 32-bit integer. """
    return _backend_op1(x, Backend.to_int32)


def to_int64(x: TensorOrTree) -> TensorOrTree:
    """ Converts the `Tensor` or `phiml.math.magic.PhiTreeNode` `x` to 64-bit integer. """
    return _backend_op1(x, Backend.to_int64)


def to_complex(x: TensorOrTree) -> TensorOrTree:
    """
    Converts the given tensor to complex floating point format with the currently specified precision.

    The precision can be set globally using `math.set_global_precision()` and locally using `with math.precision()`.

    See the documentation at https://tum-pbs.github.io/PhiML/Data_Types.html

    See Also:
        `cast()`.

    Args:
        x: values to convert

    Returns:
        `Tensor` of same shape as `x`
    """
    return _backend_op1(x, Backend.to_complex)


def is_finite(x: TensorOrTree) -> TensorOrTree:
    """ Returns a `Tensor` or `phiml.math.magic.PhiTreeNode` matching `x` with values `True` where `x` has a finite value and `False` otherwise. """
    return _backend_op1(x, Backend.isfinite)


def is_nan(x: TensorOrTree) -> TensorOrTree:
    """ Returns a `Tensor` or `phiml.math.magic.PhiTreeNode` matching `x` with values `True` where `x` is `NaN` and `False` otherwise. """
    return _backend_op1(x, Backend.isnan)


def is_inf(x: TensorOrTree) -> TensorOrTree:
    """ Returns a `Tensor` or `phiml.math.magic.PhiTreeNode` matching `x` with values `True` where `x` is `+inf` or `-inf` and `False` otherwise. """
    return _backend_op1(x, Backend.isnan)


def real(x: TensorOrTree) -> TensorOrTree:
    """
    See Also:
        `imag()`, `conjugate()`.

    Args:
        x: `Tensor` or `phiml.math.magic.PhiTreeNode` or native tensor.

    Returns:
        Real component of `x`.
    """
    return _backend_op1(x, Backend.real)


def imag(x: TensorOrTree) -> TensorOrTree:
    """
    Returns the imaginary part of `x`.
    If `x` does not store complex numbers, returns a zero tensor with the same shape and dtype as this tensor.

    See Also:
        `real()`, `conjugate()`.

    Args:
        x: `Tensor` or `phiml.math.magic.PhiTreeNode` or native tensor.

    Returns:
        Imaginary component of `x` if `x` is complex, zeros otherwise.
    """
    return _backend_op1(x, Backend.imag)


def conjugate(x: TensorOrTree) -> TensorOrTree:
    """
    See Also:
        `imag()`, `real()`.

    Args:
        x: Real or complex `Tensor` or `phiml.math.magic.PhiTreeNode` or native tensor.

    Returns:
        Complex conjugate of `x` if `x` is complex, else `x`.
    """
    return _backend_op1(x, Backend.conj)


def degrees_to_radians(deg: TensorOrTree) -> TensorOrTree:
    """ Convert degrees to radians. """
    return tree_map(lambda x: x * (3.14159265358979323846 / 180), deg)


def radians_to_degrees(rad: TensorOrTree) -> TensorOrTree:
    """ Convert degrees to radians. """
    return tree_map(lambda x: x * (180 / 3.14159265358979323846), rad)


def sin(x: TensorOrTree) -> TensorOrTree:
    """ Computes *sin(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.sin)


def arcsin(x: TensorOrTree) -> TensorOrTree:
    """ Computes the inverse of *sin(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`.
    For real arguments, the result lies in the range [-π/2, π/2].
    """
    return _backend_op1(x, Backend.arcsin)


def cos(x: TensorOrTree) -> TensorOrTree:
    """ Computes *cos(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.cos)


def arccos(x: TensorOrTree) -> TensorOrTree:
    """ Computes the inverse of *cos(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`.
    For real arguments, the result lies in the range [0, π].
    """
    return _backend_op1(x, Backend.arccos)


def tan(x: TensorOrTree) -> TensorOrTree:
    """ Computes *tan(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.tan)


def arctan(x: TensorOrTree, divide_by=None) -> TensorOrTree:
    """
    Computes the inverse of *tan(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`.

    Args:
        x: Input. The single-argument `arctan` function cannot output π/2 or -π/2 since tan(π/2) is infinite.
        divide_by: If specified, computes `arctan(x/divide_by)` so that it can return π/2 and -π/2.
            This is equivalent to the common `arctan2` function.
    """
    if divide_by is None:
        return _backend_op1(x, Backend.arctan)
    else:
        divide_by = to_float(divide_by)
        return custom_op2(x, divide_by, arctan, lambda a, b: choose_backend(a, b).arctan2(a, b), 'arctan')


def angle(x: TensorOrTree) -> TensorOrTree:
    """
    Compute the angle of a complex number.
    This is equal to *atan(Im/Re)* for most values.

    Args:
        x: `Tensor` or `phiml.math.magic.PhiTreeNode`

    Returns:
        Angle of complex number in radians.
    """
    return arctan(imag(x), divide_by=real(x))


def sinh(x: TensorOrTree) -> TensorOrTree:
    """ Computes *sinh(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.sinh)


def arcsinh(x: TensorOrTree) -> TensorOrTree:
    """ Computes the inverse of *sinh(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.arcsinh)


def cosh(x: TensorOrTree) -> TensorOrTree:
    """ Computes *cosh(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.cosh)


def arccosh(x: TensorOrTree) -> TensorOrTree:
    """ Computes the inverse of *cosh(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.arccosh)


def tanh(x: TensorOrTree) -> TensorOrTree:
    """ Computes *tanh(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.tanh)


def arctanh(x: TensorOrTree) -> TensorOrTree:
    """ Computes the inverse of *tanh(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.arctanh)


def log(x: TensorOrTree) -> TensorOrTree:
    """ Computes the natural logarithm of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.log)


def log2(x: TensorOrTree) -> TensorOrTree:
    """ Computes *log(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x` with base 2. """
    return _backend_op1(x, Backend.log2)


def log10(x: TensorOrTree) -> TensorOrTree:
    """ Computes *log(x)* of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x` with base 10. """
    return _backend_op1(x, Backend.log10)


def sigmoid(x: TensorOrTree) -> TensorOrTree:
    """ Computes the sigmoid function of the `Tensor` or `phiml.math.magic.PhiTreeNode` `x`. """
    return _backend_op1(x, Backend.sigmoid)


def cast_same(*values: Tensor) -> Tuple[Tensor]:
    """
    Casts all tensors to the same `DType`.
    If all data types are of the same kind, returns the largest occurring data type.
    Otherwise casts `bool` &rarr; `int` &rarr; `float` &rarr; `complex`.

    Args:
        *values: tensors to cast

    Returns:
        Tuple of Tensors with same data type.
    """
    assert all(isinstance(v, Tensor) for v in values), f"Only Tensor arguments allowed but got {values}"
    dtypes = [v.dtype for v in values]
    if any(dt != dtypes[0] for dt in dtypes):
        common_type = combine_types(*dtypes, fp_precision=get_precision())
        return tuple([cast(v, common_type) for v in values])
    else:
        return values


def safe_div(x: Union[float, Tensor], y: Union[float, Tensor]):
    """ Computes *x/y* with the `Tensor`s `x` and `y` but returns 0 where *y=0*. """
    return custom_op2(x, y,
                      l_operator=safe_div,
                      l_native_function=lambda x_, y_: choose_backend(x_, y_).divide_no_nan(x_, y_),
                      r_operator=lambda y_, x_: safe_div(x_, y_),
                      r_native_function=lambda y_, x_: choose_backend(x_, y_).divide_no_nan(x_, y_),
                      op_name='divide_no_nan')


def maximum(x: Union[Tensor, float], y: Union[Tensor, float]):
    """ Computes the element-wise maximum of `x` and `y`. """
    return custom_op2(x, y, maximum, lambda x_, y_: choose_backend(x_, y_).maximum(x_, y_), op_name='maximum')


def minimum(x: Union[Tensor, float], y: Union[Tensor, float]):
    """ Computes the element-wise minimum of `x` and `y`. """
    return custom_op2(x, y, minimum, lambda x_, y_: choose_backend(x_, y_).minimum(x_, y_), op_name='minimum')


def clip(x: Tensor, lower_limit: Union[float, Tensor], upper_limit: Union[float, Tensor]):
    """ Limits the values of the `Tensor` `x` to lie between `lower_limit` and `upper_limit` (inclusive). """
    if isinstance(lower_limit, Number) and isinstance(upper_limit, Number):

        def clip_(x):
            return x._op1(lambda native: choose_backend(native).clip(native, lower_limit, upper_limit))

        return broadcast_op(clip_, [x])
    else:
        return maximum(lower_limit, minimum(x, upper_limit))


def convolve(value: Tensor,
             kernel: Tensor,
             extrapolation: 'e_.Extrapolation' = None) -> Tensor:
    """
    Computes the convolution of `value` and `kernel` along the spatial axes of `kernel`.

    The channel dimensions of `value` are reduced against the equally named dimensions of `kernel`.
    The result will have the non-reduced channel dimensions of `kernel`.

    Args:
        value: `Tensor` whose shape includes all spatial dimensions of `kernel`.
        kernel: `Tensor` used as convolutional filter.
        extrapolation: If not None, pads `value` so that the result has the same shape as `value`.

    Returns:
        `Tensor`
    """
    assert all(dim in value.shape for dim in kernel.shape.spatial.names), f"Value must have all spatial dimensions of kernel but got value {value} kernel {kernel}"
    conv_shape = kernel.shape.spatial
    in_channels = value.shape.channel
    out_channels = kernel.shape.channel.without(in_channels)
    batch = value.shape.batch & kernel.shape.batch
    if extrapolation is not None and extrapolation != e_.ZERO:
        value = pad(value, {dim: (kernel.shape.get_size(dim) // 2, (kernel.shape.get_size(dim) - 1) // 2) for dim in conv_shape.names}, extrapolation)
    native_kernel = reshaped_native(kernel, (batch, out_channels, in_channels, *conv_shape.names), force_expand=in_channels)
    native_value = reshaped_native(value, (batch, in_channels, *conv_shape.names), force_expand=batch)
    backend = choose_backend(native_value, native_kernel)
    native_result = backend.conv(native_value, native_kernel, zero_padding=extrapolation == e_.ZERO)
    result = reshaped_tensor(native_result, (batch, out_channels, *conv_shape))
    return result


def boolean_mask(x, dim: DimFilter, mask: Tensor):
    """
    Discards values `x.dim[i]` where `mask.dim[i]=False`.
    All dimensions of `mask` that are not `dim` are treated as batch dimensions.

    Alternative syntax: `x.dim[mask]`.

    Implementations:

    * NumPy: Slicing
    * PyTorch: [`masked_select`](https://pytorch.org/docs/stable/generated/torch.masked_select.html)
    * TensorFlow: [`tf.boolean_mask`](https://www.tensorflow.org/api_docs/python/tf/boolean_mask)
    * Jax: Slicing

    Args:
        x: `Tensor` or `phiml.math.magic.Sliceable`.
        dim: Dimension of `x` to along which to discard slices.
        mask: Boolean `Tensor` marking which values to keep. Must have the dimension `dim` matching `x´.

    Returns:
        Selected values of `x` as `Tensor` with dimensions from `x` and `mask`.
    """
    dim, original_dim = shape(mask).only(dim), dim
    assert dim, f"mask dimension '{original_dim}' must be present on the mask {mask.shape}"
    assert dim.rank == 1, f"boolean mask only supports 1D selection"
    if not isinstance(x, Tensor) or is_sparse(x):
        keep_slices = nonzero_slices(mask)
        x_slices = [x[s] for s in keep_slices]
        return concat(x_slices, dim.name)
    
    def uniform_boolean_mask(x: Tensor, mask_1d: Tensor):
        if dim in x.shape:
            x_native = x.native(x.shape.names)  # order does not matter
            mask_native = mask_1d.native()  # only has 1 dim
            backend = choose_backend(x_native, mask_native)
            result_native = backend.boolean_mask(x_native, mask_native, axis=x.shape.index(dim))
            new_shape = x.shape.with_sizes(backend.staticshape(result_native))
            return NativeTensor(result_native, new_shape)
        else:
            total = int(sum_(to_int64(mask_1d), mask_1d.shape))
            new_shape = mask_1d.shape.with_sizes([total])
            return expand(x, new_shape)

    return broadcast_op(uniform_boolean_mask, [x, mask], iter_dims=mask.shape.without(dim))


def gather(values, indices: Tensor, dims: Union[DimFilter, None] = None):
    """
    Gathers the entries of `values` at positions described by `indices`.
    All non-channel dimensions of `indices` that are part of `values` but not indexed are treated as batch dimensions.

    See Also:
        `scatter()`.

    Args:
        values: `Tensor` or `phiml.math.matic.PhiTreeNode` containing values to gather.
        indices: `int` `Tensor`. Multidimensional position references in `values`.
            Must contain a single channel dimension for the index vector matching the number of dimensions to index.
            This channel dimension should list the dimension names to index as item names unless explicitly specified as `dims`.
        dims: (Optional) Dimensions indexed by `indices`.
            Alternatively, the dimensions can be specified as the item names of the channel dimension of `indices`.
            If `None` and no index item names are specified, will default to all spatial dimensions or all instance dimensions, depending on which ones are present (but not both).

    Returns:
        `Tensor` with combined batch dimensions, channel dimensions of `values` and spatial/instance dimensions of `indices`.
    """
    if not isinstance(values, Tensor):
        return tree_map(lambda v: gather(v, indices, dims), values)
    assert channel(indices).rank < 2, f"indices can at most have one channel dimension but got {indices.shape}"
    if dims is None:
        if channel(indices) and channel(indices).item_names[0]:
            dims = channel(indices).item_names[0]
        else:  # Fallback to spatial / instance
            warnings.warn(f"Indexing without item names is not recommended. Got indices {indices.shape}", SyntaxWarning, stacklevel=2)
            assert values.shape.instance.is_empty or values.shape.spatial.is_empty, f"Specify gather dimensions for values with both instance and spatial dimensions. Got {values.shape}"
            dims = values.shape.instance if values.shape.spatial.is_empty else values.shape.spatial
            assert dims, f"Specify gather dimensions for values with neither instance nor spatial dimensions. Got {values.shape}"
    dims = parse_dim_order(dims)
    assert dims, f"No indexing dimensions for tensor {values.shape} given indices {indices.shape}"
    if dims not in values.shape:
        return expand(values, non_channel(indices))
    if len(dims) > 1:
        assert channel(indices).rank > 0, f"indices must have a channel dimension listing the indexed dims {dims} but got {indices.shape}. You can create it via vec({', '.join([d+'=...' for d in dims])}) or channel(index='{','.join(dims)}'). If you have raveled indices, use unpack_dim(indices, channel, values.shape['{','.join(dims)}'])."
        assert channel(indices).rank == 1, f"indices must have a single channel dimension listing the indexed dims {dims} but got {indices.shape}."
    assert channel(indices).volume == len(dims), f"channel dim of indices must have size equal to the number of indexed dims {dims} but got {channel(indices)} which has {channel(indices).volume} entries"
    if indices.dtype.kind == bool:
        indices = to_int32(indices)
    if values._is_tracer or is_sparse(values):
        if not channel(indices):
            indices = expand(indices, channel(gather=dims))
        if not channel(indices).item_names[0]:
            indices = indices._with_shape_replaced(indices.shape.with_dim_size(channel(indices), dims))
        if values._is_tracer:
            return values._gather(indices)
        else:
            return sparse_gather(values, indices)
    broadcast = broadcast_dims(values, indices)
    treat_as_batch = non_channel(indices).non_instance.only(values.shape).without(dims)
    batch_ = ((values.shape.batch & indices.shape.batch).without(dims) & treat_as_batch).without(broadcast)
    channel_ = values.shape.without(dims).without(batch_).without(broadcast)

    def uniform_gather(values, indices):
        index_list_dims = indices.shape.non_channel.without(batch_)
        squeeze_index_list = False
        if not index_list_dims:
            index_list_dims = instance(_single_index=1)
            squeeze_index_list = True
        native_values = reshaped_native(values, [batch_, *dims, channel_])
        native_indices = reshaped_native(indices, [batch_, *index_list_dims, channel(indices)])
        backend = choose_backend(native_values, native_indices)
        native_result = backend.batched_gather_nd(native_values, native_indices)
        result = reshaped_tensor(native_result, [batch_, *index_list_dims, channel_], convert=False)
        if squeeze_index_list:
            result = result[{'_single_index': 0}]
        return result

    return broadcast_op(uniform_gather, [values, indices], )


def scatter(base_grid: Union[Tensor, Shape],
            indices: Union[Tensor, dict],
            values: Union[Tensor, float],
            mode: Union[str, Callable] = 'update',
            outside_handling: str = 'discard',
            indices_gradient=False,
            default=None):
    """
    Scatters `values` into `base_grid` at `indices`.
    instance dimensions of `indices` and/or `values` are reduced during scattering.
    Depending on `mode`, this method has one of the following effects:

    * `mode='update'`: Replaces the values of `base_grid` at `indices` by `values`. The result is undefined if `indices` contains duplicates.
    * `mode='add'`: Adds `values` to `base_grid` at `indices`. The values corresponding to duplicate indices are accumulated.
    * `mode='mean'`: Replaces the values of `base_grid` at `indices` by the mean of all `values` with the same index.

    Implementations:

    * NumPy: Slice assignment / `numpy.add.at`
    * PyTorch: [`torch.scatter`](https://pytorch.org/docs/stable/generated/torch.scatter.html), [`torch.scatter_add`](https://pytorch.org/docs/stable/generated/torch.scatter_add.html)
    * TensorFlow: [`tf.tensor_scatter_nd_add`](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_add), [`tf.tensor_scatter_nd_update`](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update)
    * Jax: [`jax.lax.scatter_add`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scatter_add.html), [`jax.lax.scatter`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scatter.html)

    See Also:
        `gather()`.

    Args:
        base_grid: `Tensor` into which `values` are scattered.
        indices: `Tensor` of n-dimensional indices at which to place `values`.
            Must have a single channel dimension with size matching the number of spatial dimensions of `base_grid`.
            This dimension is optional if the spatial rank is 1.
            Must also contain all `scatter_dims`.
        values: `Tensor` of values to scatter at `indices`.
        mode: Scatter mode as `str` or function.
            Supported modes are 'add', 'mean', 'update', 'max', 'min', 'prod', 'any', 'all'.
            The corresponding functions are the built-in `sum`, `max´, `min`, as well as the reduce functions in `phiml.math`.
        outside_handling: Defines how indices lying outside the bounds of `base_grid` are handled.

            * `'discard'`: outside indices are ignored.
            * `'clamp'`: outside indices are projected onto the closest point inside the grid.
            * `'undefined'`: All points are expected to lie inside the grid. Otherwise an error may be thrown or an undefined tensor may be returned.
        indices_gradient: Whether to allow the gradient of this operation to be backpropagated through `indices`.
        default: Default value to use for bins into which no value is scattered.
            By default, `NaN` is used for the modes `update` and `mean`, `0` for `sum`, `inf` for min and `-inf` for max.
            This will upgrade the data type to `float` if necessary.

    Returns:
        Copy of `base_grid` with updated values at `indices`.
    """
    if callable(mode):
        mode = {sum: 'add', max: 'max', min: 'min', sum_: 'add', max_: 'max', min_: 'min', mean: 'mean', prod: 'prod', any_: 'any', all_: 'all'}[mode]
    if mode == 'prod':
        log_base_grid = log(base_grid) if isinstance(base_grid, Tensor) else base_grid
        log_default = None if default is None else log(default)
        log_result = scatter(log_base_grid, indices, log(values), 'add', outside_handling, indices_gradient, log_default)
        return exp(log_result)
    elif mode == 'any':
        b_base_grid = cast(base_grid, bool) if isinstance(base_grid, Tensor) else base_grid
        b_values = cast(values, bool)
        i_result = scatter(b_base_grid, indices, b_values, 'add', outside_handling, indices_gradient, False)
        return cast(i_result, bool)
    elif mode == 'all':
        not_base_grid = ~cast(base_grid, bool) if isinstance(base_grid, Tensor) else base_grid
        not_values = ~cast(values, bool)
        i_result = scatter(not_base_grid, indices, not_values, 'add', outside_handling, indices_gradient, False)
        return ~cast(i_result, bool)
    assert mode in ('update', 'add', 'mean', 'max', 'min'), f"Invalid scatter mode: '{mode}'"
    assert outside_handling in ('discard', 'clamp', 'undefined')
    assert isinstance(indices_gradient, bool)
    if isinstance(indices, dict):  # update a slice
        if len(indices) == 1 and isinstance(next(iter(indices.values())), (str, int, slice)):  # update a range
            dim, sel = next(iter(indices.items()))
            full_dim = base_grid.shape[dim]
            if isinstance(sel, str):
                sel = full_dim.item_names[0].index(sel)
            if isinstance(sel, int):
                sel = slice(sel, sel+1)
            assert isinstance(sel, slice), f"Selection must be a str, int or slice but got {type(sel)}"
            values = expand(values, full_dim.after_gather({dim: sel}))
            parts = [
                base_grid[{dim: slice(sel.start)}],
                values,
                base_grid[{dim: slice(sel.stop, None)}]
            ]
            return concat(parts, dim)
        else:
            raise NotImplementedError("scattering into non-continuous values not yet supported by dimension")
    grid_shape = base_grid if isinstance(base_grid, Shape) else base_grid.shape
    assert channel(indices).rank < 2
    if channel(indices) and channel(indices).item_names[0]:
        indexed_dims = channel(indices).item_names[0]
        assert indexed_dims in grid_shape, f"Scatter indices {indices.shape} point to missing dimensions in grid {grid_shape}"
        if indexed_dims != grid_shape.only(indexed_dims).names:
            indices = indices[{channel: grid_shape.only(indexed_dims).names}]
        indexed_dims = grid_shape.only(indexed_dims)
    else:
        indexed_dims = grid_shape.spatial or grid_shape.instance
        assert channel(indices).rank == 1 or (grid_shape.spatial_rank + grid_shape.instance_rank == 1 and indices.shape.channel_rank == 0), f"indices must have a channel dimension listing the indexed dims {indexed_dims} but got {indices.shape}. You can create it via vec({', '.join([d+'=...' for d in indexed_dims.names])}) or channel(index='{','.join(indexed_dims.names)}'). If you have raveled indices, use unpack_dim(indices, channel, base_grid.shape['{','.join(indexed_dims.names)}'])."
        assert channel(indices).volume == indexed_dims.rank
    values = wrap(values)
    batches = values.shape.non_channel.non_instance & indices.shape.non_channel.non_instance
    channels = grid_shape.without(indexed_dims).without(batches) & values.shape.channel
    # --- Set up grid ---
    if isinstance(base_grid, Shape):
        with choose_backend_t(indices, values):
            base_grid = zeros(base_grid & batches & values.shape.channel, dtype=values.dtype)
        if default is not None:
            base_grid += default
        elif mode in ['update', 'mean']:
            base_grid += float('nan')
        elif mode == 'max':
            base_grid -= float('inf')
        elif mode == 'min':
            base_grid += float('inf')
        else:
            assert mode == 'add'  # initialize with zeros
    # --- Handle outside indices ---
    if outside_handling == 'clamp':
        indices = clip(indices, 0, tensor(indexed_dims, channel(indices)) - 1)
    elif outside_handling == 'discard':
        indices_linear = pack_dims(indices, instance, instance(_scatter_instance=1))
        indices_inside = min_((round_(indices_linear) >= 0) & (round_(indices_linear) < tensor(indexed_dims, channel(indices_linear))), channel)
        indices_linear = boolean_mask(indices_linear, '_scatter_instance', indices_inside)
        if instance(values).rank > 0:
            values_linear = pack_dims(values, instance, instance(_scatter_instance=1))
            values_linear = boolean_mask(values_linear, '_scatter_instance', indices_inside)
            values = unpack_dim(values_linear, '_scatter_instance', instance(values))
        indices = unpack_dim(indices_linear, '_scatter_instance', instance(indices))
        if indices.shape.is_non_uniform:
            raise NotImplementedError()
    lists = indices.shape.instance & values.shape.instance

    def scatter_forward(base_grid: Tensor, indices: Tensor, values: Tensor):
        if values._is_tracer:
            if indices._is_tracer or base_grid._is_tracer:
                raise NotImplementedError("scattering linear tracer into linear tracer not supported")
            if not channel(indices):
                indices = expand(indices, channel(scatter_idx=indexed_dims))
            return values._scatter(base_grid, indices)
        indices = to_int32(round_(indices))
        native_grid = reshaped_native(base_grid, [batches, *indexed_dims, channels])
        native_values = reshaped_native(values, [batches, lists, channels])
        native_indices = reshaped_native(indices, [batches, lists, channel])
        backend = choose_backend(native_indices, native_values, native_grid)
        if mode != 'mean':
            native_result = backend.scatter(native_grid, native_indices, native_values, mode=mode)
        else:  # mean
            zero_grid = backend.zeros_like(native_grid)
            summed = backend.scatter(zero_grid, native_indices, native_values, mode='add')
            count = backend.scatter(zero_grid, native_indices, backend.ones_like(native_values), mode='add')
            native_result = summed / backend.maximum(count, 1)
            native_result = backend.where(count == 0, native_grid, native_result)
        return reshaped_tensor(native_result, [batches, *indexed_dims, channels], check_sizes=True)

    def scatter_backward(args: dict, _output, d_output):
        from ._nd import spatial_gradient
        values_grad = gather(d_output, args['indices'])
        spatial_gradient_indices = gather(spatial_gradient(d_output, dims=indexed_dims), args['indices'])
        indices_grad = mean(spatial_gradient_indices * args['values'], 'vector_')
        return None, indices_grad, values_grad

    from ._functional import custom_gradient
    scatter_function = custom_gradient(scatter_forward, scatter_backward) if indices_gradient else scatter_forward
    result = scatter_function(base_grid, indices, values)
    return result


def histogram(values: Tensor, bins: Shape or Tensor = spatial(bins=30), weights=1, same_bins: DimFilter = None):
    """
    Compute a histogram of a distribution of values.

    *Important Note:* In its current implementation, values outside the range of bins may or may not be added to the outermost bins.

    Args:
        values: `Tensor` listing the values to be binned along spatial or instance dimensions.
            `values´ may not contain channel or dual dimensions.
        bins: Either `Shape` specifying the number of equally-spaced bins to use or bin edge positions as `Tensor` with a spatial or instance dimension.
        weights: `Tensor` assigning a weight to every value in `values` that will be added to the bin, default 1.
        same_bins: Only used if `bins` is given as a `Shape`.
            Use the same bin sizes and positions across these batch dimensions.
            By default, bins will be chosen independently for each example.

    Returns:
        hist: `Tensor` containing all batch dimensions and the `bins` dimension with dtype matching `weights`.
        bin_edges: `Tensor`
        bin_center: `Tensor`
    """
    assert isinstance(values, Tensor), f"values must be a Tensor but got {type(values)}"
    assert channel(values).is_empty, f"Only 1D histograms supported but values have a channel dimension: {values.shape}"
    assert dual(values).is_empty, f"values cannot contain dual dimensions but got shape {values.shape}"
    weights = wrap(weights)
    if isinstance(bins, Shape):
        def equal_bins(v):
            return linspace(finite_min(v, shape), finite_max(v, shape), bins.with_size(bins.size + 1))
        bins = broadcast_op(equal_bins, [values], iter_dims=(batch(values) & batch(weights)).without(same_bins))
    assert isinstance(bins, Tensor), f"bins must be a Tensor but got {type(bins)}"
    assert non_batch(bins).rank == 1, f"bins must contain exactly one spatial or instance dimension listing the bin edges but got shape {bins.shape}"
    assert channel(bins).rank == dual(bins).rank == 0, f"bins cannot have any channel or dual dimensions but got shape {bins.shape}"
    tensors = [values, bins] if weights is None else [values, weights, bins]
    backend = choose_backend_t(*tensors)

    def histogram_uniform(values: Tensor, bin_edges: Tensor, weights):
        batch_dims = batch(values) & batch(bin_edges) & batch(weights)
        value_dims = non_batch(values) & non_batch(weights)
        values_native = reshaped_native(values, [batch_dims, value_dims])
        weights_native = reshaped_native(weights, [batch_dims, value_dims])
        bin_edges_native = reshaped_native(bin_edges, [batch_dims, non_batch(bin_edges)])
        hist_native = backend.histogram1d(values_native, weights_native, bin_edges_native)
        hist = reshaped_tensor(hist_native, [batch_dims, non_batch(bin_edges).with_size(non_batch(bin_edges).size - 1)])
        return hist
        # return stack_tensors([bin_edges, hist], channel(vector=[bin_edges.shape.name, 'hist']))

    bin_center = (bins[{non_batch(bins).name: slice(1, None)}] + bins[{non_batch(bins).name: slice(0, -1)}]) / 2
    bin_center = expand(bin_center, channel(vector=non_batch(bins).names))
    bin_edges = stack_tensors([bins], channel(values)) if channel(values) else bins
    return broadcast_op(histogram_uniform, [values, bins, weights]), bin_edges, bin_center


def fft(x: Tensor, dims: DimFilter = spatial) -> Tensor:
    """
    Performs a fast Fourier transform (FFT) on all spatial dimensions of x.
    
    The inverse operation is `ifft()`.

    Implementations:

    * NumPy: [`np.fft.fft`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html),
      [`numpy.fft.fft2`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html),
      [`numpy.fft.fftn`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftn.html)
    * PyTorch: [`torch.fft.fft`](https://pytorch.org/docs/stable/fft.html)
    * TensorFlow: [`tf.signal.fft`](https://www.tensorflow.org/api_docs/python/tf/signal/fft),
      [`tf.signal.fft2d`](https://www.tensorflow.org/api_docs/python/tf/signal/fft2d),
      [`tf.signal.fft3d`](https://www.tensorflow.org/api_docs/python/tf/signal/fft3d)
    * Jax: [`jax.numpy.fft.fft`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fft.fft.html),
      [`jax.numpy.fft.fft2`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fft.fft2.html)
      [`jax.numpy.fft.fft`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fft.fftn.html)

    Args:
        x: Uniform complex or float `Tensor` with at least one spatial dimension.
        dims: Dimensions along which to perform the FFT.
            If `None`, performs the FFT along all spatial dimensions of `x`.

    Returns:
        *Ƒ(x)* as complex `Tensor`
    """
    dims = x.shape.only(dims)
    x_native = x.native(x.shape)
    result_native = choose_backend(x_native).fft(x_native, x.shape.indices(dims))
    return NativeTensor(result_native, x.shape)


def ifft(k: Tensor, dims: DimFilter = spatial):
    """
    Inverse of `fft()`.

    Args:
        k: Complex or float `Tensor` with at least one spatial dimension.
        dims: Dimensions along which to perform the inverse FFT.
            If `None`, performs the inverse FFT along all spatial dimensions of `k`.

    Returns:
        *Ƒ<sup>-1</sup>(k)* as complex `Tensor`
    """
    dims = k.shape.only(dims)
    k_native = k.native(k.shape)
    result_native = choose_backend(k_native).ifft(k_native, k.shape.indices(dims))
    return NativeTensor(result_native, k.shape)


def dtype(x) -> DType:
    """
    Returns the data type of `x`.

    Args:
        x: `Tensor` or native tensor.

    Returns:
        `DType`
    """
    if isinstance(x, Tensor):
        return x.dtype
    else:
        return choose_backend(x).dtype(x)


def always_close(t1: Union[Number, Tensor, bool], t2: Union[Number, Tensor, bool], rel_tolerance=1e-5, abs_tolerance=0, equal_nan=False) -> bool:
    """
    Checks whether two tensors are guaranteed to be `close` in all values.
    Unlike `close()`, this function can be used with JIT compilation and with tensors of incompatible shapes.
    Incompatible tensors are never close.

    If one of the given tensors is being traced, the tensors are only equal if they reference the same native tensor.
    Otherwise, an element-wise equality check is performed.

    See Also:
        `close()`.

    Args:
        t1: First tensor or number to compare.
        t2: Second tensor or number to compare.
        rel_tolerance: Relative tolerance, only used if neither tensor is traced.
        abs_tolerance: Absolute tolerance, only used if neither tensor is traced.
        equal_nan: If `True`, tensors are considered close if they are NaN in the same places.

    Returns:
        `bool`
    """
    t1 = wrap(t1)
    t2 = wrap(t2)
    if t1.available != t2.available:
        return False
    if t1.available and t2.available:
        try:
            return close(t1, t2, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance, equal_nan=equal_nan)
        except IncompatibleShapes:
            return False
    elif isinstance(t1, NativeTensor) and isinstance(t2, NativeTensor):
        return t1._native is t2._native
    else:
        return t1 is t2


def close(*tensors, rel_tolerance=1e-5, abs_tolerance=0, equal_nan=False) -> bool:
    """
    Checks whether all tensors have equal values within the specified tolerance.
    
    Does not check that the shapes exactly match.
    Unlike with `always_close()`, all shapes must be compatible and tensors with different shapes are reshaped before comparing.

    See Also:
        `always_close()`.

    Args:
        *tensors: At least two  `Tensor` or tensor-like objects or `None`.
            The shapes of all tensors must be compatible but not all tensors must have all dimensions.
            If any argument is `None`, returns `True` only if all are `None`.
        rel_tolerance: Relative tolerance
        abs_tolerance: Absolute tolerance
        equal_nan: If `True`, tensors are considered close if they are NaN in the same places.

    Returns:
        `bool`, whether all given tensors are equal to the first tensor within the specified tolerance.
    """
    if tensors[0] is None:
        return all(o is None for o in tensors)
    if any(o is None for o in tensors):
        return False
    tensors = [wrap(t) for t in tensors]
    for other in tensors[1:]:
        if not _close(tensors[0], other, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance, equal_nan=equal_nan):
            return False
    return True


def equal(*objects, equal_nan=False) -> bool:
    """
    Checks whether all objects are equal.

    See Also:
        `close()`, `always_close()`.

    Args:
        *objects: Objects to compare. Can be tensors or other objects or `None`
        equal_nan: If all objects are tensor-like, whether to count `NaN` values as equal.

    Returns:
        `bool`, whether all given objects are equal to the first one.
    """
    if objects[0] is None:
        return all(o is None for o in objects)
    if any(o is None for o in objects):
        return False
    try:
        tensors = [wrap(o) for o in objects]
        if any(t.dtype.kind == object for t in tensors):
            raise ValueError
    except ValueError:  # not all are tensor-like
        return all(o == objects[0] for o in objects)
    return close(*tensors, rel_tolerance=0, abs_tolerance=0, equal_nan=equal_nan)


def _close(tensor1: Tensor, tensor2: Tensor, rel_tolerance=1e-5, abs_tolerance=0, equal_nan=False):
    if tensor2 is tensor1:
        return True
    if tensor1.shape.is_non_uniform or tensor2.shape.is_non_uniform:
        non_uniform_dims = tensor2.shape.shape.without('dims') & tensor1.shape.shape.without('dims')
        inner_close = [_close(t1, t2) for t1, t2 in zip(unstack(tensor1, non_uniform_dims), unstack(tensor2, non_uniform_dims))]
        return all(inner_close)
    new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
    np1 = choose_backend(native1).numpy(native1)
    np2 = choose_backend(native2).numpy(native2)
    return np.allclose(np1, np2, rel_tolerance, abs_tolerance, equal_nan=equal_nan)


def assert_close(*values,
                 rel_tolerance: float = 1e-5,
                 abs_tolerance: float = 0,
                 msg: str = "",
                 verbose: bool = True):
    """
    Checks that all given tensors have equal values within the specified tolerance.
    Raises an AssertionError if the values of this tensor are not within tolerance of any of the other tensors.
    
    Does not check that the shapes match as long as they can be broadcast to a common shape.

    Args:
      values: Tensors or native tensors or numbers or sequences of numbers.
      rel_tolerance: Relative tolerance.
      abs_tolerance: Absolute tolerance.
      msg: Optional error message.
      verbose: Whether to print conflicting values.
    """
    if not values:
        return
    ml_tensors = [t for t in values if isinstance(t, Tensor)]
    if ml_tensors:
        values = [compatible_tensor(t, ml_tensors[0].shape)._simplify() for t in values]  # use Tensor to infer dimensions
        for other in values[1:]:
            _assert_close(values[0], other, rel_tolerance, abs_tolerance, msg, verbose)
    elif all(isinstance(v, PhiTreeNode) for v in values):
        tree0, tensors0 = disassemble_tree(values[0], cache=False)
        for value in values[1:]:
            tree, tensors_ = disassemble_tree(value, cache=False)
            assert tree0 == tree, f"Tree structures do not match: {tree0} and {tree}"
            for t0, t in zip(tensors0, tensors_):
                _assert_close(t0, t, rel_tolerance, abs_tolerance, msg, verbose)
    else:
        np_values = [choose_backend(t).numpy(t) for t in values]
        for other in np_values[1:]:
            np.testing.assert_allclose(np_values[0], other, rel_tolerance, abs_tolerance, err_msg=msg, verbose=verbose)


def _assert_close(tensor1: Tensor, tensor2: Tensor, rel_tolerance: float, abs_tolerance: float, msg: str, verbose: bool):
    if tensor2 is tensor1:
        return
    # if isinstance(tensor2, (int, float, bool)):
    #     np.testing.assert_allclose(tensor1.numpy(), tensor2, rel_tolerance, abs_tolerance)
    if isinstance(tensor1, Layout):
        tensor1._assert_close(tensor2, rel_tolerance, abs_tolerance, msg, verbose)
    elif isinstance(tensor2, Layout):
        tensor2._assert_close(tensor1, rel_tolerance, abs_tolerance, msg, verbose)
    elif is_sparse(tensor1):
        if is_sparse(tensor2) and type(tensor2) == type(tensor1):
            _assert_close(tensor1._values, tensor2._values, rel_tolerance, abs_tolerance, msg, verbose)
            _assert_close(tensor1._indices, tensor2._indices, 0, 0, msg, verbose)
            if isinstance(tensor2, CompressedSparseMatrix) and isinstance(tensor1, CompressedSparseMatrix):
                _assert_close(tensor1._pointers, tensor2._pointers, 0, 0, msg, verbose)
        else:
            _assert_close(dense(tensor1), tensor2, rel_tolerance, abs_tolerance, msg, verbose)
    elif is_sparse(tensor2):
        return _assert_close(tensor2, tensor1, rel_tolerance, abs_tolerance, msg, verbose)
    else:
        def inner_assert_close(tensor1, tensor2):
            new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
            np1 = choose_backend(native1).numpy(native1)
            np2 = choose_backend(native2).numpy(native2)
            if not np.allclose(np1, np2, rel_tolerance, abs_tolerance):
                np.testing.assert_allclose(np1, np2, rel_tolerance, abs_tolerance, err_msg=msg, verbose=verbose)

        broadcast_op(inner_assert_close, [tensor1, tensor2], no_return=True)


def _native_wrapper(tensor_function: Callable, create_native_function: Callable, persistent_refs=False):
    INPUT_TENSORS = []
    OUTPUT_TENSORS = []

    def native_function(*natives):
        natives = list(natives)
        values = [t._op1(lambda _: natives.pop(0)) for t in INPUT_TENSORS]
        assert len(natives) == 0, "Not all arguments were converted"
        result = tensor_function(*values)
        results = [result] if not isinstance(result, (tuple, list)) else result
        OUTPUT_TENSORS.clear()
        OUTPUT_TENSORS.extend(results)
        return sum([v._natives() for v in results], ())

    backend = default_backend()
    traced = create_native_function(native_function, backend)
    if traced is NotImplemented:
        warnings.warn(f"Backend '{backend}' not supported. Returning original function.", RuntimeWarning)
        return tensor_function, None, INPUT_TENSORS, OUTPUT_TENSORS

    def wrapper(*values: Tensor):
        INPUT_TENSORS.clear()
        INPUT_TENSORS.extend(values)
        values = [cached(v) for v in values]
        natives = sum([v._natives() for v in values], ())
        results_native = list(traced(*natives))
        results = [t._with_natives_replaced(results_native) for t in OUTPUT_TENSORS]
        if not persistent_refs:
            INPUT_TENSORS.clear()
            # OUTPUT_TENSORS.clear()  outputs need to be saved because native_function may be called only the first time. Will get garbage collected once the function is not referenced anymore.
        assert len(results_native) == 0
        return results[0] if len(results) == 1 else results

    return wrapper, traced, INPUT_TENSORS, OUTPUT_TENSORS


def stop_gradient(x):
    """
    Disables gradients for the given tensor.
    This may switch off the gradients for `x` itself or create a copy of `x` with disabled gradients.

    Implementations:

    * PyTorch: [`x.detach()`](https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach)
    * TensorFlow: [`tf.stop_gradient`](https://www.tensorflow.org/api_docs/python/tf/stop_gradient)
    * Jax: [`jax.lax.stop_gradient`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.stop_gradient.html)

    Args:
        x: `Tensor` or `phiml.math.magic.PhiTreeNode` for which gradients should be disabled.

    Returns:
        Copy of `x`.
    """
    return _backend_op1(x, Backend.stop_gradient)


def pairwise_distances(positions: Tensor,
                       max_distance: Union[float, Tensor] = None,
                       format: Union[str, Tensor] = 'dense',
                       default: Optional[float] = None,
                       method: str = 'sparse') -> Tensor:
    """
    Computes the distance matrix containing the pairwise position differences between each pair of points.
    Points that are further apart than `max_distance` (if specified) are assigned a distance value of `0`.
    The diagonal of the matrix (self-distance) also consists purely of zero-vectors and may or may not be stored explicitly.

    Args:
        positions: `Tensor`.
            Channel dimensions are interpreted as position components.
            Instance and spatial dimensions list nodes.
        max_distance: Scalar or `Tensor` specifying a max_radius for each point separately.
            Can contain additional batch dimensions but spatial/instance dimensions must match `positions` if present.
            If not specified, uses an infinite cutoff radius, i.e. all points will be considered neighbors.
        format: Matrix format as `str` or concrete sparsity pattern as `Tensor`.
            Allowed strings are `'dense', `'csr'`, `'coo'`, `'csc'`.
            When a `Tensor` is passed, it needs to have all instance and spatial dims as `positions` as well as corresponding dual dimensions.
            The distances will be evaluated at all stored entries of the `format` tensor.
        default: Value the sparse tensor returns for non-stored values. Must be `0` or `None`.

    Returns:
        Distance matrix as sparse or dense `Tensor`, depending on `format`.
        For each spatial/instance dimension in `positions`, the matrix also contains a dual dimension of the same name and size.
        The matrix also contains all batch dimensions of `positions` and one channel dimension called `vector`.

    Examples:
        >>> pos = vec(x=0, y=tensor([0, 1, 2.5], instance('particles')))
        >>> dx = pairwise_distances(pos, format='dense', max_distance=2)
        >>> dx.particles[0]
        (x=0.000, y=0.000); (x=0.000, y=1.000); (x=0.000, y=0.000) (~particlesᵈ=3, vectorᶜ=x,y)
    """
    assert isinstance(positions, Tensor), f"positions must be a Tensor but got {type(positions)}"
    assert default in [0, None], f"default value must be either 0 or None but got '{default}'"
    primal_dims = positions.shape.non_batch.non_channel.non_dual
    dual_dims = primal_dims.as_dual()
    # --- Dense ---
    if (isinstance(format, str) and format == 'dense') or (isinstance(format, Tensor) and get_format(format) == 'dense'):
        if isinstance(format, Tensor):
            dual_dims = dual(format)
        dx = unpack_dim(pack_dims(positions, non_batch(positions).non_channel.non_dual, instance('_tmp')), '_tmp', dual_dims) - positions
        if max_distance is not None:
            neighbors = sum_(dx ** 2, channel) <= max_distance ** 2
            default = float('nan') if default is None else default
            dx = where(neighbors, dx, default)
        return dx
    # --- sparse with known connectivity ---
    if isinstance(format, Tensor):  # sparse connectivity specified, no neighborhood search required
        assert max_distance is None, "max_distance not allowed when connectivity is specified (passing a Tensor for format)"
        assert is_sparse(format)
        return map_pairs(lambda p1, p2: p2 - p1, positions, format)
    # --- Sparse neighbor search ---
    assert max_distance is not None, "max_distance must be specified when computing distance in sparse format"
    max_distance = wrap(max_distance)
    index_dtype = DType(int, 32)
    backend = choose_backend_t(positions, max_distance)
    batch_shape = batch(positions) & batch(max_distance)
    if not dual_dims.well_defined:
        assert dual_dims.rank == 1, f"others_dims sizes must be specified when passing more then one dimension but got {dual_dims}"
        dual_dims = dual_dims.with_size(primal_dims.volume)
    # --- Determine mode ---
    # if method == 'sklearn':

    pair_count = None
    table_len = None
    mode = 'vectorize' if batch_shape.volume > 1 and batch_shape.is_uniform else 'loop'
    if backend.is_available(positions):
        if mode == 'vectorize':
            # ToDo determine limits from positions? build_cells+bincount would be enough
            pair_count = 7
    else:  # tracing
        if backend.requires_fixed_shapes_when_tracing():
            # ToDo use fixed limits (set by user)
            pair_count = 7
            mode = 'vectorize'
    # --- Run neighborhood search ---
    from ..backend._partition import find_neighbors_sparse, find_neighbors_semi_sparse, find_neighbors_matscipy, find_neighbors_sklearn
    if mode == 'loop':
        indices = []
        values = []
        for b in batch_shape.meshgrid():
            native_positions = reshaped_native(positions[b], [primal_dims, channel(positions)])
            native_max_dist = max_distance[b].native()
            if method == 'sparse':
                nat_rows, nat_cols, nat_vals = find_neighbors_sparse(native_positions, native_max_dist, None, periodic=False, default=default)
            elif method == 'semi-sparse':
                nat_rows, nat_cols, nat_vals, req_pair_count, req_max_occupancy = find_neighbors_semi_sparse(native_positions, native_max_dist, None, periodic=False, default=default)
            elif method == 'matscipy':
                assert positions.available, f"Cannot jit-compile matscipy neighborhood search"
                nat_rows, nat_cols, nat_vals = find_neighbors_matscipy(native_positions, native_max_dist, None, periodic=False)
            elif method == 'sklearn':
                assert positions.available, f"Cannot jit-compile matscipy neighborhood search"
                nat_rows, nat_cols, nat_vals = find_neighbors_sklearn(native_positions, native_max_dist)
            else:
                raise ValueError(method)
            nat_indices = backend.stack([nat_rows, nat_cols], -1)
            indices.append(reshaped_tensor(nat_indices, [instance('pairs'), channel(vector=primal_dims.names + dual_dims.names)], convert=False))
            values.append(reshaped_tensor(nat_vals, [instance('pairs'), channel(positions)]))
        indices = stack(indices, batch_shape)
        values = stack(values, batch_shape)
    elif mode == 'vectorize':
        raise NotImplementedError
        # native_positions = reshaped_native(positions, [batch_shape, primal_dims, channel(positions)])
        # native_max_dist = reshaped_native(max_distance, [batch_shape, primal_dims], force_expand=False)
        # def single_search(pos, r):
        #     return find_neighbors(pos, r, None, periodic=False, pair_count=pair_count, default=default)
        # nat_rows, nat_cols, nat_vals = backend.vectorized_call(single_search, native_positions, native_max_dist, output_dtypes=(index_dtype, index_dtype, positions.dtype))
        # nat_indices = backend.stack([nat_rows, nat_cols], -1)
        # indices = reshaped_tensor(nat_indices, [batch_shape, instance('pairs'), channel(vector=primal_dims.names + dual_dims.names)], convert=False)
        # values = reshaped_tensor(nat_vals, [batch_shape, instance('pairs'), channel(positions)])
    else:
        raise RuntimeError
    # --- Assemble sparse matrix ---
    dense_shape = primal_dims & dual_dims
    coo = SparseCoordinateTensor(indices, values, dense_shape, can_contain_double_entries=False, indices_sorted=False, default=default)
    return to_format(coo, format)


def map_pairs(map_function: Callable, values: Tensor, connections: Tensor):
    """
    Evaluates `map_function` on all pairs of elements present in the sparsity pattern of `connections`.

    Args:
        map_function: Function with signature `(Tensor, Tensor) -> Tensor`.
        values: Values to evaluate `map_function` on.
            Needs to have a spatial or instance dimension but must not have a dual dimension.
        connections: Sparse tensor.

    Returns:
        `Tensor` with the sparse dimensions of `connections` and all non-instance dimensions returned by `map_function`.
    """
    assert dual(values).is_empty, f"values must not have a dual dimension but got {values.shape}"
    indices = stored_indices(connections, invalid='clamp')
    origin_dim, neighbors_dim = channel(indices).item_names[0]
    if origin_dim not in values.shape:
        origin_dim, neighbors_dim = neighbors_dim, origin_dim
    assert origin_dim in values.shape, f"No dimension of connections {connections.shape} is present in values {values.shape}"
    origin = values[{origin_dim: indices[origin_dim]}]
    target = values[{origin_dim: indices[neighbors_dim]}]
    result = map_function(origin, target)
    return tensor_like(connections, result, value_order='as existing')
