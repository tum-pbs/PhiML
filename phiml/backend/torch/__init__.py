"""
PyTorch integration.
"""
from ._torch_backend import TorchBackend as _TorchBackend

TORCH = _TorchBackend()
""" Backend for PyTorch operations. """

__all__ = [key for key in globals().keys() if not key.startswith('_')]
