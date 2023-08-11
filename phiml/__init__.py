"""
Open-source ...


Project homepage: https://github.com/tum-pbs/PhiML

Documentation overview: https://tum-pbs.github.io/PhiML

PyPI: https://pypi.org/project/phiml/
"""

import os as _os


with open(_os.path.join(_os.path.dirname(__file__), 'VERSION'), 'r') as version_file:
    __version__ = version_file.read()


def verify():
    """
    Checks your configuration for potential problems and prints a summary.

    To run verify without importing `phiml`, run the script `tests/verify.py` included in the source distribution.
    """
    import sys
    from ._troubleshoot import assert_minimal_config, troubleshoot
    try:
        assert_minimal_config()
    except AssertionError as fail_err:
        print("\n".join(fail_err.args), file=sys.stderr)
        return
    print(troubleshoot())


def set_logging_level(level='debug'):
    """
    Sets the logging level for Φ-ML functions.

    Args:
        level: Logging level, one of `'critical', 'fatal', 'error', 'warning', 'info', 'debug'`
    """
    from .backend import ML_LOGGER
    ML_LOGGER.setLevel(level.upper())
