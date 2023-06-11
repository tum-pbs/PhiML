"""
Jax integration.
"""
from ._jax_backend import JaxBackend as _JaxBackend

JAX = _JaxBackend()
"""Backend for Jax operations."""

# _math.backend.BACKENDS.append(JAX)

__all__ = [key for key in globals().keys() if not key.startswith('_')]
