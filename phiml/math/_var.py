from typing import TypeVar, Any, Tuple

from ._magic_ops import tree_map, variable_attributes
from ._tensors import Tensor, wrap
from ._ops import convert
from ..backend import default_backend

Obj = TypeVar('Obj')

def module_from_variables(obj: Obj, backend=None) -> Tuple[Obj, Any]:
    backend = backend if backend is not None else default_backend()
    variables = []
    def leaf_param(x: Tensor):
        x = convert(x, backend)
        var = backend.variable(x.native(x.shape))
        variables.append(var)
        return wrap(var, x.shape)
    tree = tree_map(leaf_param, obj, attr_type=variable_attributes)
    module = backend.module({f'param_{i}': p for i, p in enumerate(variables)})
    return tree, module
