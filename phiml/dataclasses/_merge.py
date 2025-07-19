import dataclasses
from functools import cached_property
from typing import Sequence, Union, Callable

from ..math import Tensor
from ..math._magic_ops import stack
from ..math._shape import Shape, NotCompatible
from ._dataclasses import replace, is_data_field

def merge_rule(field_name: str, rule: Union[str, Callable] = 'default', simplify=True):
    def wrap(cls):
        assert dataclasses.is_dataclass(cls), f"@merge_rule can only be used on dataclasses."
        names = [f.name for f in dataclasses.fields(cls)]
        assert field_name in names, f"@merge_rule defined for nonexistent field '{field_name}'"
        if not hasattr(cls, '__merge_rules__'):
            cls.__merge_rules__ = {}
        cls.__merge_rules__[field_name] = {'rule': rule, 'simplify': simplify}
        return cls
    return wrap


DEFAULT_RULE = {'rule': 'default', 'simplify': True}
ATTRS_RULE = {'rule': 'join', 'simplify': True}


def get_merge_rule(cls, field_name: str):
    if hasattr(cls, '__merge_rules__') and field_name in cls.__merge_rules__:
        return cls.__merge_rules__[field_name]
    return ATTRS_RULE if field_name in ('variable_attrs', 'value_attrs') else DEFAULT_RULE


# def merge(*values, field_name: str, is_attribute: bool, rule: Union[str, Callable], simplify: bool):
#     """
#     Returns:
#         Sequence of length 1 to indicate that all values map to the same
#         Or sequence of same length as `values´.
#     """
#     if callable(rule):
#         result = rule(*values)
#         return result
#     elif rule == 'default':
#         if is_attribute:
#             stack([v.center for v in values], dim, simplify=True, **kwargs)
#         else:
#             # default for other fields: all-equal...?
#     elif rule == 'join':
#         result = set().update(*values)
#         return [result]
#     elif rule == 'all-equal':
#         if all(v == values[0] for v in values[1:]):
#             return [values[0]]
#         return NotImplemented


def dc_stack(objs: Sequence, dim: Shape, expand_values=False, simplify=False, layout_non_matching=False, **kwargs):
    if any(type(o) != type(objs[0]) for o in objs):
        raise NotImplementedError  # fallback stack
    updates = {}
    for f in dataclasses.fields(objs[0]):
        if f.name == 'variable_attrs':
            updates[f.name] = tuple(set.union(*[set(obj.variable_attrs) for obj in objs]))
        elif f.name == 'value_attrs':
            va0 = set(objs[0].value_attrs)
            assert all(set(obj.value_attrs) == va0 for obj in objs), f"value_attrs must match among all stacked instances of {type(objs[0]).__name__} but got {[obj.value_attrs for obj in objs]}"
        else:
            is_attribute = is_data_field(f)
            if is_attribute:
                values = [getattr(obj, f.name) for obj in objs]
                updates[f.name] = stack(values, dim, expand_values=expand_values, simplify=simplify, layout_non_matching=True)
            else:
                value0 = getattr(objs[0], f.name)
                try:
                    configs_equal = all([getattr(obj, f.name) == value0 for obj in objs])
                except ValueError as err:
                    raise IncompatibleDataclassConfigs(f"Attribute '{f.name}' comparison failed for '{objs}'", err)
                if not configs_equal:
                    raise IncompatibleDataclassConfigs(f"Attribute '{f.name}' must match among all stacked objects")

    result = replace(objs[0], **updates)
    # --- stack cache ---
    c_names = set(k for k in objs[0].__dict__ if isinstance(getattr(type(objs[0]), k, None), cached_property))
    shared = c_names & set.intersection(*[set(obj.__dict__) for obj in objs])
    for key in shared:
        values = [getattr(obj, key) for obj in objs]
        if all(isinstance(v, Tensor) for v in values):
            result.__dict__[key] = stack(values, dim, expand_values=expand_values, simplify=simplify, layout_non_matching=layout_non_matching, **kwargs)
    return result


def dc_concat(obj: Sequence, dim):
    raise NotImplementedError  # concat cached properties


def dc_expand(obj, dims):
    raise NotImplementedError  # expand cached Tensor properties


class IncompatibleDataclassConfigs(NotCompatible): ...
