import traceback
import weakref
from typing import Sequence, Dict, Optional

import h5py

from ..backend._backend import get_backend
from ..math._tensors import Dense, Layout, TensorStack
from .. import Shape, DType
from ..backend import Backend, ML_LOGGER


class H5Source:
    """
    Represents an opened HDF5 file.
    Instances can be pickled and unpickled. When unpickled, the file will only be opened once accessed again.
    Many Tensors can reference the same H5Source in order to avoid unnecessary file openings.
    Compared to an h5py.File caching system, this system automatically cleans up opened files once all references to it are garbage collected.
    """

    def __init__(self, path: str):
        self.path = path
        self._handle: Optional[h5py.File] = None

    @property
    def h5file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, 'r')
        return self._handle

    def __getstate__(self):
        return {'path': self.path}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._handle = None


class DiskTensor(Dense):

    def __init__(self, source: H5Source, array_name: str, array_slices: Dict[str, slice], names: Sequence[str], expanded_shape: Shape, backend: Backend, dtype: DType):
        # Do not call super __init__, because we override _native as a property
        self.source = source
        self.array_name = array_name
        self.array_slices = array_slices
        self._memory_tensor = None  # weakref, not pickled
        self._shape = expanded_shape
        self._names = names
        self._backend = backend
        self._dtype = dtype

    @property
    def _native(self):
        if self._memory_tensor is not None:
            if isinstance(self._memory_tensor, weakref.ReferenceType):
                memory_tensor = self._memory_tensor()
            else:
                memory_tensor = self._memory_tensor
            if memory_tensor is not None:
                return memory_tensor
        ML_LOGGER.debug(f"Loading {self.source.path} into memory")
        # traceback.print_stack()
        try:
            file = self.source.h5file
            data_ref = file[self.array_name]
        except Exception as e:
            traceback.print_exception(e)
            raise e
        if not self.array_slices:
            data = data_ref[:] if self._names else data_ref[()]
        else:
            ordered_slices = [self.array_slices.get(n, slice(None)) for n in self._names]
            data = data_ref[ordered_slices]
        try:
            self._memory_tensor = weakref.ref(data)
        except TypeError:
            self._memory_tensor = data  # primitives (such as NumPy scalar types) don't support weak references. Store them directly.
        return data

    @property
    def available(self) -> bool:
        if self._memory_tensor is None:
            return False
        if isinstance(self._memory_tensor, weakref.ReferenceType):
            return False  # even if currently in memory, this may change at the whim of the GC at any time
        return True

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_memory_tensor'] = None
        state['_backend'] = self._backend.name
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._memory_tensor = None
        self._backend = get_backend(self._backend)

    @property
    def dtype(self) -> DType:
        return self._dtype

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **_kwargs) -> 'Tensor':
        if any(isinstance(v, Layout) for v in values):
            layout_ = [v for v in values if isinstance(v, Layout)][0]
            return layout_.__stack__(values, dim, **_kwargs)
        return TensorStack(values, dim)
