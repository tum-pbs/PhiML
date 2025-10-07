import ast
import inspect
from typing import Set


def is_jupyter_notebook() -> bool:
    """ Check if code is running in a Jupyter notebook environment. """
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is None:
            return False
        return 'IPKernelApp' in ipython.config  # Check if it's a notebook (not just IPython terminal)
    except (ImportError, AttributeError):
        return False


def is_class_from_notebook(cls) -> bool:
    """ Detect if a class was defined in a Jupyter notebook. """
    if not is_jupyter_notebook():
        return False
    module = inspect.getmodule(cls)
    if module is None:
        return False
    n = module.__name__
    return n in {'__main__', '<stdin>', '<console>'} or n.startswith('IPython')


def class_to_string(cls) -> str:
    """
    Convert a Python class to a self-contained string with imports.

    Args:
        cls: The class to convert

    Returns:
        A string containing all imports and the class definition

    Example:
        >>> class MyClass:
        ...     def method(self):
        ...         return np.array([1, 2, 3])
        >>> code = class_to_string(MyClass)
        >>> exec(code)  # Re-creates the class
    """
    imports = _get_class_imports(cls)
    imports = '\n'.join(sorted(imports))
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError) as e:
        raise ValueError(f"Cannot get source for class {cls.__name__}: {e}")
    source = inspect.cleandoc(source)  # Remove leading indentation
    return f"""
{imports}

{source}
"""


def _get_class_imports(cls) -> Set[str]:
    """Extract import statements needed for a class."""
    imports = set()
    module = inspect.getmodule(cls)
    if module is None:
        return imports
    # --- Get class source and parse it ---
    try:
        source = inspect.getsource(cls)
        tree = ast.parse(source)
    except (OSError, TypeError):
        return imports
    # --- Find all names used in the class ---
    used_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Get the root name (e.g., 'np' from 'np.array')
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                used_names.add(root.id)
    # --- Check module namespace for these names ---
    if hasattr(module, '__dict__'):
        for name in used_names:
            if name in module.__dict__:
                obj = module.__dict__[name]
                obj_module = inspect.getmodule(obj)
                # Skip builtins and local definitions
                if obj_module is None or obj_module.__name__ == '__main__':
                    continue
                # Generate import statement
                if hasattr(obj, '__name__'):
                    if obj.__name__ == name:
                        # Direct import: from module import name
                        imports.add(f"from {obj_module.__name__} import {name}")
                    else:
                        # Aliased import
                        imports.add(f"from {obj_module.__name__} import {obj.__name__} as {name}")
                elif obj_module.__name__ == name:
                    # Module import: import module
                    imports.add(f"import {name}")
    return imports
