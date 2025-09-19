import ast
import inspect
import textwrap
from typing import Dict, Set


class PropertySplitter(ast.NodeTransformer):
    """AST transformer to split property methods into multiple properties."""

    def __init__(self, method_name: str):
        self.method_name = method_name
        self.local_vars: Dict[str, ast.stmt] = {}
        self.var_references: Set[str] = set()
        self.return_expr: ast.expr = None

    def visit_Assign(self, node: ast.Assign) -> ast.stmt:
        """Collect assignment statements to local variables."""
        # Only handle simple assignments to single targets
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            # Skip if it's an attribute assignment or other complex case
            if not var_name.startswith('_') and var_name != 'self':
                self.local_vars[var_name] = node
                self.var_references.add(var_name)
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
        """Handle augmented assignments (+=, -=, etc.)."""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            if not var_name.startswith('_') and var_name != 'self':
                self.local_vars[var_name] = node
                self.var_references.add(var_name)
        return node

    def visit_Return(self, node: ast.Return) -> ast.stmt:
        """Capture the return expression."""
        if node.value:
            self.return_expr = node.value
        return node

    def visit_Name(self, node: ast.Name) -> ast.expr:
        """Track variable references."""
        if isinstance(node.ctx, ast.Load) and node.id in self.local_vars:
            self.var_references.add(node.id)
        return node


def split_property_into_methods(property_obj) -> str:
    """
    Takes a property object and splits it into multiple property methods.

    Args:
        property_obj: A property object (e.g., MyClass.my_property)

    Returns:
        str: Generated code with multiple property methods
    """

    # Get the property's getter function
    if not isinstance(property_obj, property):
        raise ValueError("Input must be a property object")

    getter_func = property_obj.fget
    if getter_func is None:
        raise ValueError("Property must have a getter function")

    # Get the source code
    try:
        source = inspect.getsource(getter_func)
    except OSError:
        raise ValueError("Cannot retrieve source code for the property")

    # Clean up the source code
    source = textwrap.dedent(source)

    # Parse the AST
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"Failed to parse source code: {e}")

    # Find the function definition
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break

    if func_def is None:
        raise ValueError("No function definition found in source code")

    method_name = func_def.name

    # Analyze the function
    splitter = PropertySplitter(method_name)
    splitter.visit(func_def)

    # Generate the new property methods
    generated_code = []

    # Create properties for local variables
    var_to_property = {}
    for var_name, assign_node in splitter.local_vars.items():
        if var_name in splitter.var_references:
            prop_name = f"generated_{method_name}_{var_name}"
            var_to_property[var_name] = prop_name

            # Create the property method
            if isinstance(assign_node, ast.Assign):
                value_expr = assign_node.value
            elif isinstance(assign_node, ast.AugAssign):
                # For augmented assignment, we need to reference the original variable
                # This is a simplified approach - in practice, you might need more sophisticated handling
                value_expr = assign_node.value
            else:
                continue

            # Replace variable references in the expression
            replacer = VariableReplacer(var_to_property)
            new_value_expr = replacer.visit(value_expr)

            # Generate property code
            prop_code = f"""    @property
    def {prop_name}(self):
        return {ast.unparse(new_value_expr)}"""
            generated_code.append(prop_code)

    # Create the main return property
    if splitter.return_expr:
        main_prop_name = f"generated_{method_name}"
        replacer = VariableReplacer(var_to_property)
        new_return_expr = replacer.visit(splitter.return_expr)

        main_prop_code = f"""    @property
    def {main_prop_name}(self):
        return {ast.unparse(new_return_expr)}"""
        generated_code.append(main_prop_code)

    return "\n\n".join(generated_code)


class VariableReplacer(ast.NodeTransformer):
    """Replace variable references with property access."""

    def __init__(self, var_to_property: Dict[str, str]):
        self.var_to_property = var_to_property

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if isinstance(node.ctx, ast.Load) and node.id in self.var_to_property:
            # Replace with property access: self.property_name
            return ast.Attribute(
                value=ast.Name(id='self', ctx=ast.Load()),
                attr=self.var_to_property[node.id],
                ctx=ast.Load()
            )
        return node


def add_method_to_class(cls, method_code: str):
    """
    Dynamically adds a method defined by `method_code` to the class `cls`.

    Args:
        cls: The class object to inject the method into.
        method_code: A string containing a valid Python method definition.
                     It should include 'self' if it's an instance method.
    """
    local_ns = {}  # Temporary namespace for execution
    exec(method_code, cls.__module__.__dict__ if hasattr(cls.__module__, "__dict__") else globals(), local_ns)
    func_name = next(iter(local_ns))  # assumes only one function in the string
    func = local_ns[func_name]
    setattr(cls, func_name, func)


def add_properties_to_class(cls, generated_code: str):
    """
    Dynamically add properties to a class by executing the generated code.

    Args:
        cls: The class to modify
        generated_code: String containing the property definitions
    """
    tree = ast.parse(generated_code)
    # Extract each property definition and add it to the class
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            for decorator in node.decorator_list:  # Check if it's a property decorator
                if (isinstance(decorator, ast.Name) and decorator.id == 'property') or \
                        (isinstance(decorator, ast.Attribute) and decorator.attr == 'property'):
                    prop_name = node.name
                    func_code = ast.unparse(node)
                    local_namespace = {}
                    exec(func_code, cls.__module__.__dict__, local_namespace)
                    func = local_namespace[prop_name]
                    prop = property(func)
                    setattr(cls, prop_name, prop)
                    break


if __name__ == "__main__":
    from phiml import wrap, Tensor, sin, isum

    class ExampleClass:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        @property
        def complex_calculation(self):
            # This property will be split into multiple properties
            a = isum(self.x * 2)
            b = sin(self.y ** 2)
            sum_result = a + b
            return sum_result * 3

    result = split_property_into_methods(ExampleClass.complex_calculation)
    print(result)

    # from breakpoints import detect_breakpoints_in_property
    # print(detect_breakpoints_in_property(ExampleClass.complex_calculation))
