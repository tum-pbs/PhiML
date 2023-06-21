# Contributing to UnifyML
All contributions are welcome!
You can mail the developers or get in touch on GitHub.


## Types of contributions we're looking for

We're open to all kind of contributions that improve or extend the UnifyML library.
We especially welcome

- Bug fixes
- Code optimizations or native (CUDA) implementations.
- Integrations with other computing libraries, such as Numba.
- Unit tests


## How to Contribute
We recommend you to contact the developers before starting your contribution.
There may already be similar internal work or planned changes that would affect how to code the contribution.

To contribute code, fork UnifyML on GitHub, make your changes, and submit a pull request.
Make sure that your contribution passes all tests.


## Style Guide
Style guidelines make the code more uniform and easier to read.
Generally we stick to the Python style guidelines as outlined in [PEP 8](https://www.python.org/dev/peps/pep-0008/), with some minor modifications outlined below.

Have a look at the [Zen](https://en.wikipedia.org/wiki/Zen_of_Python) [of Python](https://www.python.org/dev/peps/pep-0020/) for the philosophy behind the rules.
We would like to add the rule *Concise is better than repetitive.*

We use PyLint for static code analysis with specific configuration files for the
[tests](../tests/.pylintrc) and the
[code base](../unifyml/.pylintrc).
PyLint is part of the automatic testing pipeline.
The warning log can be viewed online by selecting a `Tests` job and expanding the pylint output.

### Docstrings
The [API documentation](https://holl-.github.io/UnifyML/) for UnifyML is generated using [pdoc](https://pdoc3.github.io/pdoc/).
We use [Google style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
with Markdown formatting.

```python
"""
Function description.

*Note in italic.*

Example:
    
    >>> def python_example()
    # Out: ...

Args:
    arg1: Description.
        Indentation for multi-line description.

Returns:
    Single output. For multi-output use same format as for args.
"""
```

Docstrings for variables are located directly below the public declaration.
```python
variable: type = value
""" Docstring for the variable. """
```


### Additional style choices
- **No line length limit**; long lines are allowed if the code remains easy to understand.
- **Code comments** should only describe information that is not obvious from the code. They should be used sparingly as the code should be understandable by itself. For documentation, use docstrings instead. Code comments that explain a single line of code should go in the same line as the code they refer to, if possible.
- Code comments that describe multiple lines precede the block and have the format `# --- Comment ---`.
- Avoid empty lines inside of methods. To separate code blocks use multi-line comments as described above.
- Use the apostrophe character ' to enclose strings that affect the program, such as different modes of a function, e.g. `f(mode='fast')`. Use double quotes for file names and text that is displayed, such as error messages, warnings, and user interface text.
