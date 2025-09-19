import ast
import dataclasses
import inspect
import types
from functools import cached_property
from typing import Set

from ..backend import ML_LOGGER


class MemberVariableAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.member_vars = set()

    def visit_Attribute(self, node):
        # Check if the attribute is accessed via `self`
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            self.member_vars.add(node.attr)
        self.generic_visit(node)


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


# @dataclasses.dataclass(frozen=True)
# class PropertyInfo


def all_field_deps(cls: type, cls_property) -> Set[str]:
    """Find all dataclass fields the specified property requires for evaluation (directly and indirectly)"""
    if hasattr(cls, '__all_field_deps__') and cls_property in cls.__all_field_deps__:
        return cls.__all_field_deps__[cls_property]
    if isinstance(cls_property, cached_property):
        method = cls_property.func
    elif isinstance(cls_property, property):
        method = cls_property.fget
    else:
        assert callable(cls_property) and hasattr(cls_property, '__qualname__'), f"Dependency resolver failed on {cls_property} of {cls.__name__}"
        method = cls_property
    ML_LOGGER.debug(f"Analyzing dependencies of {method.__qualname__}")
    source_code = inspect.getsource(method)
    indent0 = len(source_code) - len(source_code.lstrip())
    source_code_top = "\n".join([line[indent0:] for line in source_code.split("\n")])
    tree = ast.parse(source_code_top)
    analyzer = MemberVariableAnalyzer()
    analyzer.visit(tree)
    direct_deps = analyzer.member_vars
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
    # --- Cache result ---
    if not hasattr(cls, '__all_field_deps__'):
        cls.__all_field_deps__ = {cls_property: field_deps}
    else:
        cls.__all_field_deps__[cls_property] = field_deps
    return field_deps


def field_deps(cls: type, cls_property: property) -> Set[str]:
    """Find dataclass fields the specified property requires for evaluation, not going through cached properties."""
    if hasattr(cls, '__field_deps__') and cls_property in cls.__field_deps__:
        return cls.__field_deps__[cls_property]
    if isinstance(cls_property, cached_property):
        method = cls_property.func
    elif isinstance(cls_property, property):
        method = cls_property.fget
    else:
        assert callable(cls_property) and hasattr(cls_property, '__qualname__'), f"Dependency resolver failed on {cls_property} of {cls.__name__}"
        method = cls_property
    ML_LOGGER.debug(f"Analyzing dependencies of {method.__qualname__}")
    source_code = inspect.getsource(method)
    indent0 = len(source_code) - len(source_code.lstrip())
    source_code_top = "\n".join([line[indent0:] for line in source_code.split("\n")])
    tree = ast.parse(source_code_top)
    analyzer = MemberVariableAnalyzer()
    analyzer.visit(tree)
    direct_deps = analyzer.member_vars
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
    # --- Cache result ---
    if not hasattr(cls, '__field_deps__'):
        cls.__field_deps__ = {cls_property: field_deps_}
    else:
        cls.__field_deps__[cls_property] = field_deps_
    return field_deps_


def cache_deps(cls: type, cls_property) -> Set[str]:
    """Lists all cached properties required by `cls_property` (direct and indirect but stopping at cached properties)"""
    if hasattr(cls, '__cache_deps__') and cls_property in cls.__cache_deps__:
        return cls.__cache_deps__[cls_property]
    if isinstance(cls_property, cached_property):
        method = cls_property.func
    elif isinstance(cls_property, property):
        method = cls_property.fget
    else:
        assert callable(cls_property) and hasattr(cls_property, '__qualname__'), f"Dependency resolver failed on {cls_property} of {cls.__name__}"
        method = cls_property
    ML_LOGGER.debug(f"Analyzing dependencies of {method.__qualname__}")
    source_code = inspect.getsource(method)
    indent0 = len(source_code) - len(source_code.lstrip())
    source_code_top = "\n".join([line[indent0:] for line in source_code.split("\n")])
    tree = ast.parse(source_code_top)
    analyzer = MemberVariableAnalyzer()
    analyzer.visit(tree)
    direct_deps = analyzer.member_vars
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
    # --- Cache result ---
    if not hasattr(cls, '__cache_deps__'):
        cls.__cache_deps__ = {cls_property: prop_deps}
    else:
        cls.__cache_deps__[cls_property] = prop_deps
    return prop_deps