"""
Open-source ...


Project homepage: https://github.com/holl-/UnifyML

Documentation overview: https://holl-.github.io/UnifyML

PyPI: https://pypi.org/project/unifyml/
"""

import os as _os


with open(_os.path.join(_os.path.dirname(__file__), 'VERSION'), 'r') as version_file:
    __version__ = version_file.read()


def verify():
    """
    Checks your configuration for potential problems and prints a summary.

    To run verify without importing `unifyml`, run the script `tests/verify.py` included in the source distribution.
    """
    import sys
    from ._troubleshoot import assert_minimal_config, troubleshoot
    try:
        assert_minimal_config()
    except AssertionError as fail_err:
        print("\n".join(fail_err.args), file=sys.stderr)
        return
    print(troubleshoot())


def detect_backends() -> tuple:
    """
    Registers all available backends and returns them.
    This includes only backends for which the minimal requirements are fulfilled.

    Returns:
        `tuple` of `unifyml.backend.Backend`
    """
    try:
        from .jax import JAX
    except ImportError:
        pass
    try:
        from .torch import TORCH
    except ImportError:
        pass
    try:
        from .tf import TENSORFLOW
    except ImportError:
        pass
    from .backend import BACKENDS
    return tuple([b for b in BACKENDS if b.name != 'Python'])


def set_logging_level(level='debug'):
    """
    Sets the logging level for PhiFlow functions.

    Args:
        level: Logging level, one of `'critical', 'fatal', 'error', 'warning', 'info', 'debug'`
    """
    from .backend import ML_LOGGER
    ML_LOGGER.setLevel(level.upper())
