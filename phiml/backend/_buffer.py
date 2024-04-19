from typing import Sequence, Tuple, Callable, Any, Dict, Optional

from ._backend import Backend, choose_backend, TensorType


_REQUIRED_SIZES: Dict[str, TensorType] = {}
_CURRENT_TRACE: Dict[str, int] = {}
_SET_CONFIG: Dict[str, int] = {}


def set_buffer_config(config: Dict[str, int]):
    """
    Call this function before tracing a new configuration.
    If this is the first trace, pass an empty dict for `config` and retrieve the default config using `get_used_config()`.

    Args:
        config:

    Returns:

    """
    _REQUIRED_SIZES.clear()
    _CURRENT_TRACE.clear()
    _SET_CONFIG.clear()
    _SET_CONFIG.update({k: int(v) for k, v in config.items()})


def get_buffer_config() -> Dict[str, int]:
    """
    Returns:
        The buffer sizes used during the current trace.
        If the trace is not completed yet, the result might be incomplete.
    """
    return dict(_CURRENT_TRACE)


def get_required_buffer_sizes() -> Dict[str, TensorType]:
    """
    Returns:
        Size placeholders / tracers.
    """
    return dict(_REQUIRED_SIZES)


def register_buffer(name: str, min_size: TensorType, default_size: int) -> int:
    """
    This function is used by low-level code to register internal buffers.

    Args:
        name: Buffer name.
        min_size: Required buffer size. This can be a placeholder / tracer.
        default_size: Size to use when no other information is available.

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
    while f"{name}{i}" in _REQUIRED_SIZES:
        i += 1
    buffer_id = f"{name}{i}"
    # --- Register buffer ---
    _REQUIRED_SIZES[buffer_id] = min_size
    if _SET_CONFIG:
        assert buffer_id in _SET_CONFIG, f"Buffer {buffer_id} was not registered during previous traces"
        _CURRENT_TRACE[buffer_id] = _SET_CONFIG[buffer_id]
        return _SET_CONFIG[buffer_id]
    else:  # no config available, use default
        _CURRENT_TRACE[buffer_id] = default_size
        return default_size


def register_buffer_deferred(name: str, dependent_tensors: Sequence[TensorType], default_size: int) -> Tuple[Optional[int], Callable]:
    """
    Register a buffer without having a placeholder for its required size yet.
    The required size needs to be provided later

    Args:
        name: Buffer name.
        dependent_tensors: All relevant tensors that can influence the required size of the buffer.
        default_size: Size to use when no other information is available.

    Returns:
        size: Buffer size to use this time.
        provide_size_fn: Function to call to provide the required size placeholder of the buffer
    """
    b = choose_backend(*dependent_tensors)
    if all(b.is_available(t) for t in dependent_tensors):
        return None, lambda s: None
    if b.name in ['torch', 'tensorflow']:
        return None, lambda s: None  # PyTorch & TF allow variable shapes during tracing. Jax doesn't
    # --- determine id ---
    i = 0
    while f"{name}{i}" in _REQUIRED_SIZES:
        i += 1
    buffer_id = f"{name}{i}"
    # --- Register buffer ---
    def provide_size(min_size: TensorType):
        _REQUIRED_SIZES[buffer_id] = min_size
    if _SET_CONFIG:
        assert buffer_id in _SET_CONFIG, f"Buffer {buffer_id} was not registered during previous traces"
        _CURRENT_TRACE[buffer_id] = _SET_CONFIG[buffer_id]
        return _SET_CONFIG[buffer_id], provide_size
    else:  # no config available, use default
        _CURRENT_TRACE[buffer_id] = default_size
        return default_size, provide_size


def wasted_memory(config: Dict[str, int], required: Dict[str, int]):
    """
    Args:
        config: Buffer size configuration.
        required: Minimum required buffer sizes. Must have same keys as `config`.

    Returns:
        If all buffers in `config` are large enough for `required`, returns the total number of unnecessary elements.
        Otherwise, i.e. if `config` is not sufficient, returns a negative number representing the total number of missing elements.
    """
    assert config.keys() == required.keys(), f"Buffers must match but got config={config}, required={required}"
    if any(config[k] < req for k, req in required.items()):  # buffers too small!
        return sum([min(0, config[k] - req) for k, req in required.items()])
    return sum([max(0, config[k] - req) for k, req in required.items()])
