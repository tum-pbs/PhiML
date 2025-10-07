import dataclasses
import dis
import types
from functools import cached_property
from typing import Set, Tuple

from ..backend import ML_LOGGER


def get_unchanged_cache(obj, updated_fields: Set[str]):
    if not hasattr(obj, '__dict__'):
        return {}
    cache = {k: v for k, v in obj.__dict__.items() if isinstance(getattr(type(obj), k, None), cached_property)}
    result = {}
    for k, v in cache.items():
        deps = all_field_deps(obj.__class__, getattr(type(obj), k, None))
        if deps.isdisjoint(updated_fields):
            result[k] = v
    return result


def get_method(cls: type, cls_property):
    if isinstance(cls_property, cached_property):
        return cls_property.func
    elif isinstance(cls_property, property):
        return cls_property.fget
    else:
        assert callable(cls_property) and hasattr(cls_property, '__qualname__'), f"Dependency resolver failed on {cls_property} of {cls.__name__}"
        return cls_property


def cache_in_class(f):
    cache_name = f'__{f.__name__}__'
    def cached(cls, cls_property):
        if hasattr(cls, cache_name) and cls_property in getattr(cls, cache_name):
            return getattr(cls, cache_name)[cls_property]
        result = f(cls, cls_property)
        if not hasattr(cls, cache_name):
            setattr(cls, cache_name, {cls_property: result})
        else:
            getattr(cls, cache_name)[cls_property] = result
        return result
    return cached


@cache_in_class
def all_field_deps(cls: type, cls_property) -> Set[str]:
    """Find all dataclass fields the specified property requires for evaluation (directly and indirectly)"""
    method = get_method(cls, cls_property)
    ML_LOGGER.debug(f"Analyzing dependencies of {method.__qualname__} (all_field_deps)")
    direct_deps = get_self_attrs_from_bytecode(method)
    fields = set([f.name for f in dataclasses.fields(cls)])
    # --- Indirect dependencies via method/property calls ---
    field_deps = direct_deps & fields
    for prop_dep in direct_deps - fields:
        if not hasattr(cls, prop_dep):  # may be a dynamic dim, such as vector
            if hasattr(cls, 'shape'):
                prop_dep = 'shape'
            else:
                field_deps.update(fields)  # automatic shape() depends on all data_fields
                continue
        if isinstance(getattr(cls, prop_dep), (property, cached_property)):
            field_deps.update(all_field_deps(cls, getattr(cls, prop_dep)))
        elif callable(getattr(cls, prop_dep)):
            field_deps.update(all_field_deps(cls, getattr(cls, prop_dep)))
    return field_deps


@cache_in_class
def field_deps(cls: type, cls_property: property) -> Set[str]:
    """Find dataclass fields the specified property requires for evaluation, not going through cached properties."""
    method = get_method(cls, cls_property)
    ML_LOGGER.debug(f"Analyzing dependencies of {method.__qualname__} (field_deps)")
    direct_deps = get_self_attrs_from_bytecode(method)
    fields = set([f.name for f in dataclasses.fields(cls)])
    field_deps_ = direct_deps & fields
    # --- Indirect dependencies via method/property calls ---
    all_caches = set([m for m in dir(cls) if isinstance(getattr(cls, m), cached_property)])
    prop_deps = all_caches & direct_deps
    for prop_dep in direct_deps - fields - prop_deps:
        if not hasattr(cls, prop_dep):  # may be a dynamic dim, such as vector
            if hasattr(cls, 'shape'):
                prop_dep = 'shape'
            else:
                field_deps_.update(fields)  # automatic shape() depends on all data_fields
                continue
        if isinstance(getattr(cls, prop_dep), property):
            field_deps_.update(field_deps(cls, getattr(cls, prop_dep)))
        elif isinstance(getattr(cls, prop_dep), cached_property):
            pass  # stop tracing on cached properties
        elif callable(getattr(cls, prop_dep)):
            field_deps_.update(field_deps(cls, getattr(cls, prop_dep)))
    return field_deps_


@cache_in_class
def cache_deps(cls: type, cls_property) -> Set[str]:
    """Lists all cached properties required by `cls_property` (direct and indirect but stopping at cached properties)"""
    method = get_method(cls, cls_property)
    ML_LOGGER.debug(f"Analyzing dependencies of {method.__qualname__} (cache_deps)")
    direct_deps = get_self_attrs_from_bytecode(method)
    fields = set([f.name for f in dataclasses.fields(cls)])
    all_caches = set([m for m in dir(cls) if isinstance(getattr(cls, m), cached_property)])
    # --- Indirect dependencies via method/property calls ---
    prop_deps = all_caches & direct_deps
    for method_dep in direct_deps - fields - prop_deps:
        if not hasattr(cls, method_dep):
            continue  # may be a dynamic dim, such as vector
        dep = getattr(cls, method_dep)
        if callable(dep) or isinstance(dep, property):  # excluding cached_property, parallel_property
            prop_deps.update(cache_deps(cls, dep))
    return prop_deps


def get_self_attrs_from_bytecode(func) -> Tuple[Set[str], Set[str]]:
    assert isinstance(func, (types.FunctionType, types.MethodType))
    code = func.__code__
    attrs = set()
    instructions = list(dis.get_instructions(code))
    for inst1, inst2 in zip(instructions[:-1], instructions[1:]):
        # Detect: LOAD_FAST self → LOAD_ATTR name
        if inst1.opname == "LOAD_FAST" and inst1.argval == "self":
            if inst2.opname in {"LOAD_ATTR", "LOAD_METHOD"}:
                attrs.add(inst2.argval)
    return attrs
