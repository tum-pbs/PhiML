"""
TensorFlow integration.
"""
import platform as _platform

import os
import tensorflow as _tf

from .. import ML_LOGGER as _LOGGER

if _tf.__version__.startswith('1.'):
    raise ImportError(f"Φ-ML requires TensorFlow 2 but found TensorFlow {_tf.__version__}")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only errors
if _platform.system().lower() == 'windows':  # prevent Blas GEMM launch failed on Windows
    for i, device in enumerate(_tf.config.list_physical_devices('GPU')):
        try:
            _tf.config.experimental.set_memory_growth(device, True)
            _LOGGER.debug(f"phiml.backend.tf: Setting memory_growth on GPU {i} to True to prevent Blas errors")
        except RuntimeError:
            _LOGGER.debug(f"phiml.backend.tf: Failed to set memory_growth on GPU {i}")

from ._compile_cuda import compile_cuda_ops

from ._tf_backend import TFBackend as _TFBackend

TENSORFLOW = _TFBackend()
"""Backend for TensorFlow operations."""

__all__ = [key for key in globals().keys() if not key.startswith('_')]
