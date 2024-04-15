from typing import Sequence, Tuple, Callable, Any, Dict

from ._backend import Backend, choose_backend, TensorType


_REQUIRED_SIZES: Dict[str, Any] = {}
_CURRENT_TRACE: Dict[str, int] = {}
_SET_CONFIG: Dict[str, int] = {}


def set_config(config: Dict[str, int]):
    """
    Call this function before tracing a new configuration.
    If this is the first trace, pass an empty dict for `config` and retrieve the default config using `get_used_config()`.

    Args:
        config:

    Returns:

    """
    _SET_CONFIG.clear()
    _SET_CONFIG.update(config)


def get_used_config() -> Dict[str, int]:
    return dict(_CURRENT_TRACE)


def get_required_size_tracers() -> Dict[str, TensorType]:
    return dict(_REQUIRED_SIZES)


def register_buffer(name: str, min_size: TensorType, default_size: int) -> int:
    """
    This function is used by low-level code to register internal buffers.

    Args:
        name: Buffer name.
        min_size: Required buffer size. This can be a placeholder / tracer.
        default_size:

    Returns:
        Buffer size as `int`.
    """
    b = choose_backend(min_size)
    if b.is_available(min_size):
        return int(min_size)
    if b.name in ['torch', 'tensorflow']:
        return min_size  # PyTorch & TF allow variable shapes during tracing. Jax doesn't
    # --- determine id ---
    i = 0
    while f"{name}_{i}" in _REQUIRED_SIZES:
        i += 1
    buffer_id = f"{name}_{i}"
    # --- Register buffer ---
    _REQUIRED_SIZES[buffer_id] = min_size
    if _SET_CONFIG:
        assert buffer_id in _SET_CONFIG, f"Buffer {buffer_id} was not registered during previous traces"
        _CURRENT_TRACE[buffer_id] = _SET_CONFIG[buffer_id]
        return _SET_CONFIG[buffer_id]
    else:  # no config available, use default
        _CURRENT_TRACE[buffer_id] = default_size
        return default_size
