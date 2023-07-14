from ._jax_backend import JaxBackend
"""Backend for Jax operations."""

JAX = JaxBackend()

__all__ = [key for key in globals().keys() if not key.startswith('_')]
