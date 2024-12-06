import ast
import inspect
from functools import cached_property
from typing import Set

from phiml.backend import ML_LOGGER


def get_unchanged_cache(obj, updated_fields: Set[str]):
    cache = {k: v for k, v in obj.__dict__.items() if isinstance(getattr(type(obj), k, None), cached_property)}
    result = {}
    for k, v in cache.items():
        deps = get_dependencies(obj.__class__, getattr(type(obj), k, None))
        if deps.isdisjoint(updated_fields):
            result[k] = v
    return result


def get_dependencies(cls: type, cls_property) -> Set[str]:
    if hasattr(cls, '__phiml_dep__') and cls_property in cls.__phiml_dep__:
        return cls.__phiml_dep__[cls_property]
    if isinstance(cls_property, cached_property):
        method = cls_property.func
    else:
        method = cls_property
    ML_LOGGER.debug(f"Analyzing dependencies of {method.__qualname__}")
    source_code = inspect.getsource(method)
    indent0 = len(source_code) - len(source_code.lstrip())
    source_code_top = "\n".join([line[indent0:] for line in source_code.split("\n")])
    tree = ast.parse(source_code_top)
    analyzer = MemberVariableAnalyzer()
    analyzer.visit(tree)
    result = analyzer.member_vars
    if not hasattr(cls, '__phiml_dep__'):
        cls.__phiml_dep__ = {cls_property: result}
    else:
        cls.__phiml_dep__[cls_property] = result
    return result


class MemberVariableAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.member_vars = set()

    def visit_Attribute(self, node):
        # Check if the attribute is accessed via `self`
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            self.member_vars.add(node.attr)
        self.generic_visit(node)
