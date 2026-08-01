# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: infer_types=True
"""
Cython-accelerated hot paths for phiml.math._shape.

Provides:
  cy_pure_merge(shapes, allow_varying_sizes, allow_varying_labels)
  cy_merge_shapes(objs, allow_varying_sizes, allow_varying_labels)
  cy_shape(obj, allow_unshaped=False)

These are drop-in replacements for the Python implementations with the same
semantics.  Only the common case (no conflicts, already-normalized inputs) is
tightly optimised; all error paths fall back transparently.
"""

# ---------------------------------------------------------------------------
# Bootstrap: import everything we need from the Python _shape module.
# We delay the module-level import to avoid a circular dependency during
# module initialisation.
# ---------------------------------------------------------------------------

cdef object _Dim        # phiml.math._shape.Dim
cdef object _PureShape  # phiml.math._shape.PureShape
cdef object _MixedShape # phiml.math._shape.MixedShape
cdef object _EMPTY_SHAPE
cdef object _IncompatibleShapes
cdef object _size_equal
cdef str    _BATCH_DIM
cdef str    _DUAL_DIM
cdef str    _INSTANCE_DIM
cdef str    _SPATIAL_DIM
cdef str    _CHANNEL_DIM

cdef bint _initialized = False


cdef void _ensure_init():
    global _Dim, _PureShape, _MixedShape, _EMPTY_SHAPE, _IncompatibleShapes
    global _size_equal
    global _BATCH_DIM, _DUAL_DIM, _INSTANCE_DIM, _SPATIAL_DIM, _CHANNEL_DIM
    global _initialized
    if _initialized:
        return
    from phiml.math._shape import (
        Dim, PureShape, MixedShape, EMPTY_SHAPE, IncompatibleShapes, _size_equal,
        BATCH_DIM, DUAL_DIM, INSTANCE_DIM, SPATIAL_DIM, CHANNEL_DIM,
    )
    _Dim              = Dim
    _PureShape        = PureShape
    _MixedShape       = MixedShape
    _EMPTY_SHAPE      = EMPTY_SHAPE
    _IncompatibleShapes = IncompatibleShapes
    _size_equal       = _size_equal
    _BATCH_DIM        = BATCH_DIM
    _DUAL_DIM         = DUAL_DIM
    _INSTANCE_DIM     = INSTANCE_DIM
    _SPATIAL_DIM      = SPATIAL_DIM
    _CHANNEL_DIM      = CHANNEL_DIM
    _initialized = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

cdef inline bint _is_dim(object s):
    return type(s) is _Dim

cdef inline bint _is_pureshape(object s):
    return type(s) is _PureShape

cdef inline bint _is_mixed(object s):
    return type(s) is _MixedShape

cdef inline bint _shape_bool(object s):
    """Fast bool(shape) without calling __bool__ via Python dispatch."""
    if type(s) is _Dim:
        return True          # Dim is always truthy
    # PureShape and MixedShape are truthy iff dims dict is non-empty
    return bool((<object>s).dims)


cdef object _make_pure_shape(str dim_type, object dims):
    """Construct a PureShape or return the single Dim if rank==1."""
    if len(dims) == 1:
        return next(iter(dims.values()))
    return _PureShape(dim_type, dims)


# ---------------------------------------------------------------------------
# cy_pure_merge
# ---------------------------------------------------------------------------

def cy_pure_merge(object shapes, bint allow_varying_sizes, bint allow_varying_labels):
    """
    Cython-accelerated implementation of pure_merge.

    Parameters
    ----------
    shapes : tuple of (Dim | PureShape)
    allow_varying_sizes : bool
    allow_varying_labels : bool
    """
    _ensure_init()

    cdef int n
    cdef object first_non_empty
    cdef int non_empty_count
    cdef object dims
    cdef object first_dim_type
    cdef object incompatible_sizes
    cdef object incompatible_labels
    cdef object s, dim
    cdef str name
    cdef object prev_dim
    cdef object names1, names2
    cdef object dims_to_merge
    cdef object sizes_match
    cdef object s1, s2

    n = len(shapes)

    # ---- binary short-circuit ----
    if n == 2:
        s1 = shapes[0]
        s2 = shapes[1]
        if not _shape_bool(s1):
            return s2
        if not _shape_bool(s2):
            return s1
        if s1 is s2:
            return s1

    # ---- main merge loop ----
    first_non_empty = None
    non_empty_count = 0
    dims = {}
    first_dim_type = None
    incompatible_sizes = None
    incompatible_labels = None

    for s in shapes:
        if type(s) is _Dim:
            if first_non_empty is None:
                first_non_empty = s
            non_empty_count += 1
            dims_to_merge = (s,)
        elif type(s) is _PureShape:
            if not (<object>s).dims:
                continue
            if first_non_empty is None:
                first_non_empty = s
            non_empty_count += 1
            dims_to_merge = (<object>s).dims.values()
        else:
            assert type(s) is _PureShape, f"Only Dim and PureShape allowed in pure_merge() but got {type(s)}"
            continue

        for dim in dims_to_merge:
            name = (<object>dim).name
            if name not in dims:
                dims[name] = dim
                if first_dim_type is None:
                    first_dim_type = (<object>dim).dim_type
            else:
                prev_dim = dims[name]
                sizes_match = _size_equal((<object>dim).size, (<object>prev_dim).size)
                if not sizes_match:
                    if allow_varying_sizes:
                        if incompatible_sizes is None:
                            incompatible_sizes = {name}
                        else:
                            incompatible_sizes.add(name)
                    else:
                        raise _IncompatibleShapes(
                            f"Cannot merge shapes {shapes} because dimension '{name}' exists with different sizes.",
                            *shapes
                        )
                else:
                    names1 = (<object>prev_dim).slice_names
                    names2 = (<object>dim).slice_names
                    if names1 is not None and names2 is not None and len(names1) > 1:
                        if names1 != names2:
                            if allow_varying_labels:
                                if incompatible_labels is None:
                                    incompatible_labels = {name}
                                else:
                                    incompatible_labels.add(name)
                            elif set(names1) == set(names2):
                                raise _IncompatibleShapes(
                                    f"Inconsistent component order on {name}: "
                                    f"'{','.join(names1)}' vs '{','.join(names2)}' "
                                    f"in dimension '{name}'. Failed to merge shapes {shapes}",
                                    *shapes
                                )
                            else:
                                raise _IncompatibleShapes(
                                    f"Cannot merge shapes {shapes} because dimension '{name}' exists with different labels.",
                                    *shapes
                                )
                    elif names1 is None and names2 is not None:
                        dims[name] = dim

    if not dims:
        return _EMPTY_SHAPE
    if non_empty_count == 1:
        return first_non_empty

    if incompatible_labels is not None:
        for name in incompatible_labels:
            dims[name] = (<object>dims[name]).with_size((<object>dims[name]).size, keep_labels=False)
    if incompatible_sizes is not None:
        for name in incompatible_sizes:
            dims[name] = (<object>dims[name]).without_sizes()

    return _make_pure_shape(first_dim_type, dims)


# ---------------------------------------------------------------------------
# cy_merge_shapes  (public fast-path dispatcher)
# ---------------------------------------------------------------------------

def cy_shape(object obj, bint allow_unshaped=False):
    """
    Fast Cython implementation of shape().
    Returns the shape of obj for Tensor/Shaped objects, or returns obj if it's already a Shape.
    Falls back to Python implementation for complex cases.
    """
    _ensure_init()
    # Fast path: already a Shape type
    if type(obj) is _Dim or type(obj) is _PureShape or type(obj) is _MixedShape:
        return obj
    
    # Check if it has a .shape attribute that's a Shape type
    cdef object obj_shape
    if hasattr(obj, 'shape'):
        obj_shape = obj.shape
        if type(obj_shape) is _Dim or type(obj_shape) is _PureShape or type(obj_shape) is _MixedShape:
            return obj_shape
    
    # Check if it has a __shape__ method
    if hasattr(obj, '__shape__'):
        return (<object>obj).__shape__()
    
    # Fallback to the original Python implementation for complex logic.
    # Importing the accelerated module-level shape wrapper would recurse back
    # into cy_shape() when this extension is enabled.
    from phiml.math._shape import _shape_orig
    return _shape_orig(obj, allow_unshaped=allow_unshaped)


cdef inline object _as_shape(object obj):
    """
    Fast Cython helper for _as_shape.
    Returns obj if it's a Shape type (Dim, PureShape, MixedShape),
    otherwise falls back to full cy_shape processing.
    """
    cdef object obj_shape
    if type(obj) is _Dim or type(obj) is _PureShape or type(obj) is _MixedShape:
        return obj
    # For non-Shape objects, use the full cy_shape function to handle all cases
    # We can't call cy_shape from an inline function safely, so we do a simple fallback
    if hasattr(obj, 'shape'):
        obj_shape = obj.shape
        if type(obj_shape) is _Dim or type(obj_shape) is _PureShape or type(obj_shape) is _MixedShape:
            return obj_shape
    if hasattr(obj, '__shape__'):
        return (<object>obj).__shape__()
    # For anything else, fall back to the original Python implementation.
    from phiml.math._shape import _shape_orig
    return _shape_orig(obj, allow_unshaped=False)


def cy_merge_shapes(object objs, bint allow_varying_sizes, bint allow_varying_labels):
    """
    Cython-accelerated implementation of merge_shapes.

    Parameters
    ----------
    objs : tuple of (Shape | Any)
    allow_varying_sizes : bool
    allow_varying_labels : bool
    """
    _ensure_init()

    # hoist ALL cdef declarations to function scope (Cython requirement)
    cdef int n
    cdef bint a_mixed, b_mixed, is_pure
    cdef object a, b, s, obj
    cdef object ba, bb, da, db, ia, ib, sa, sb, ca, cb
    cdef object b_, d_, i_, s_, c_
    cdef object bm, dm, im, sm, cm
    cdef object dims2, dimsN
    cdef object shapes
    cdef object batch_s, dual_s, instance_s, spatial_s, channel_s
    cdef object common_dim_type

    n = len(objs)

    if n == 0:
        return _EMPTY_SHAPE

    if allow_varying_sizes:
        allow_varying_labels = True

    if n == 1:
        return _as_shape(objs[0])

    # ---- fast binary path ----
    if n == 2:
        a = _as_shape(objs[0])
        b = _as_shape(objs[1])
        if not _shape_bool(a):
            return b
        if not _shape_bool(b):
            return a
        if a is b:
            return a
        a_mixed = _is_mixed(a)
        b_mixed = _is_mixed(b)
        if (not a_mixed) and (not b_mixed):
            if (<object>a).dim_type == (<object>b).dim_type:
                return cy_pure_merge((a, b), allow_varying_sizes, allow_varying_labels)
            # Two pure shapes, different types — cheap manual split
            ba = (<object>a).batch if (<object>a).dim_type == _BATCH_DIM else _EMPTY_SHAPE
            bb = (<object>b).batch if (<object>b).dim_type == _BATCH_DIM else _EMPTY_SHAPE
            b_ = cy_pure_merge((ba, bb), allow_varying_sizes, allow_varying_labels)
            da = (<object>a).dual if (<object>a).dim_type == _DUAL_DIM else _EMPTY_SHAPE
            db = (<object>b).dual if (<object>b).dim_type == _DUAL_DIM else _EMPTY_SHAPE
            d_ = cy_pure_merge((da, db), allow_varying_sizes, allow_varying_labels)
            ia = (<object>a).instance if (<object>a).dim_type == _INSTANCE_DIM else _EMPTY_SHAPE
            ib = (<object>b).instance if (<object>b).dim_type == _INSTANCE_DIM else _EMPTY_SHAPE
            i_ = cy_pure_merge((ia, ib), allow_varying_sizes, allow_varying_labels)
            sa = (<object>a).spatial if (<object>a).dim_type == _SPATIAL_DIM else _EMPTY_SHAPE
            sb = (<object>b).spatial if (<object>b).dim_type == _SPATIAL_DIM else _EMPTY_SHAPE
            s_ = cy_pure_merge((sa, sb), allow_varying_sizes, allow_varying_labels)
            ca = (<object>a).channel if (<object>a).dim_type == _CHANNEL_DIM else _EMPTY_SHAPE
            cb = (<object>b).channel if (<object>b).dim_type == _CHANNEL_DIM else _EMPTY_SHAPE
            c_ = cy_pure_merge((ca, cb), allow_varying_sizes, allow_varying_labels)
            dims2 = {}
            if _shape_bool(b_): dims2.update((<object>b_).dims)
            if _shape_bool(d_): dims2.update((<object>d_).dims)
            if _shape_bool(i_): dims2.update((<object>i_).dims)
            if _shape_bool(s_): dims2.update((<object>s_).dims)
            if _shape_bool(c_): dims2.update((<object>c_).dims)
            return _MixedShape(b_, d_, i_, s_, c_, dims2) if dims2 else _EMPTY_SHAPE
        # At least one MixedShape — fall through to N-ary path with 2 shapes
        shapes = [a, b]
    else:
        # ---- N-ary path ----
        shapes = []
        is_pure = True
        common_dim_type = None
        for obj in objs:
            s = _as_shape(obj)
            shapes.append(s)
            if _is_mixed(s):
                is_pure = False
            elif _shape_bool(s):
                if common_dim_type is None:
                    common_dim_type = (<object>s).dim_type
                elif (<object>s).dim_type != common_dim_type:
                    is_pure = False
        if is_pure:
            return cy_pure_merge(tuple(shapes), allow_varying_sizes, allow_varying_labels)

    # ---- Mixed N-ary assembly ----
    batch_s = []
    dual_s = []
    instance_s = []
    spatial_s = []
    channel_s = []
    for s in shapes:
        batch_s.append((<object>s).batch)
        dual_s.append((<object>s).dual)
        instance_s.append((<object>s).instance)
        spatial_s.append((<object>s).spatial)
        channel_s.append((<object>s).channel)
    bm = cy_pure_merge(tuple(batch_s),    allow_varying_sizes, allow_varying_labels)
    dm = cy_pure_merge(tuple(dual_s),     allow_varying_sizes, allow_varying_labels)
    im = cy_pure_merge(tuple(instance_s), allow_varying_sizes, allow_varying_labels)
    sm = cy_pure_merge(tuple(spatial_s),  allow_varying_sizes, allow_varying_labels)
    cm = cy_pure_merge(tuple(channel_s),  allow_varying_sizes, allow_varying_labels)
    dimsN = {}
    if _shape_bool(bm): dimsN.update((<object>bm).dims)
    if _shape_bool(dm): dimsN.update((<object>dm).dims)
    if _shape_bool(im): dimsN.update((<object>im).dims)
    if _shape_bool(sm): dimsN.update((<object>sm).dims)
    if _shape_bool(cm): dimsN.update((<object>cm).dims)
    return _MixedShape(bm, dm, im, sm, cm, dimsN) if dimsN else _EMPTY_SHAPE



