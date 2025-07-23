from dataclasses import dataclass
from functools import lru_cache
from typing import Union

import numpy as np
import sys


class DTypeMeta(type):

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs:
            return {bool: BOOL, object: OBJECT}[args[0]]
        if len(args) == 2 and not kwargs:
            return DType.by_bits(*args)
        elif 'precision' in kwargs:
            assert len(args) == 1, "If 'precision' is specified, only one argument may be given."
            return DType.by_precision(args[0], kwargs['precision'])
        return type.__call__(self, *args, **kwargs)


@dataclass(frozen=True)
class DType(metaclass=DTypeMeta):
    """
    Instances of `DType` represent the kind and size of data elements.
    The data type of tensors can be obtained via `Tensor.dtype`.

    The following kinds of data types are supported:

    * `float` with 32 / 64 bits
    * `complex` with 64 / 128 bits
    * `int` with 8 / 16 / 32 / 64 bits
    * `bool` with 8 bits
    * `str` with 8*n* bits

    Unlike with many computing libraries, there are no global variables corresponding to the available types.
    Instead, data types can simply be instantiated as needed.
    """
    kind: type
    """Python type, one of `(bool, int, float, complex, str, object)`"""
    bits: int
    """Number of bits per element, typically a multiple of 8."""
    unsigned: bool
    """If `True`, the data type is unsigned, meaning it can only represent non-negative values."""
    exponent_bits: int
    """Number of bits used for the exponent in floating point types. 0 for integers."""
    mantissa_bits: int
    """Number of bits used for the mantissa in floating point types. Same as `bits` for integers."""
    finite_only: bool
    """If `True`, the data type can only represent finite values, i.e., no NaN or Inf."""
    unsigned_zero: bool
    """If `True`, the data type cannot represent signed zeros. This is `True` for integers and `False` for most floating point types."""

    @property
    def precision(self):
        """ Floating point precision. Only defined if `kind in (float, complex)`. For complex values, returns half of `DType.bits`. """
        if self.kind == float:
            return self.bits
        if self.kind == complex:
            return self.bits // 2
        else:
            return None

    @property
    def itemsize(self):
        """ Number of bytes used to storea single value of this type. See `DType.bits`. """
        return self.bits // 8 if self.bits % 8 == 0 else self.bits / 8

    def __eq__(self, other):
        if isinstance(other, DType):
            return self.kind == other.kind and self.bits == other.bits and self.unsigned == other.unsigned
        elif other in {bool, int, float, complex, object}:
            return self.kind == other
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.kind)

    def __repr__(self):
        if self.kind == int:
            if self.unsigned:
                return f"uint{self.bits}"
            else:
                return f"int{self.bits}"
        elif self.kind == float:
            if self == FLOAT16:
                return "float16"
            elif self == FLOAT32:
                return "float32"
            elif self == FLOAT64:
                return "float64"
            else:
                return f"float{self.bits}_e{self.exponent_bits}_m{self.mantissa_bits}{'fn' if self.finite_only else ''}{'uz' if self.unsigned_zero else ''}"
        elif self.kind == complex:
            if self == COMPLEX64:
                return "complex64"
            elif self == COMPLEX128:
                return "complex128"
            else:
                return f"complex{self.bits}_e{self.exponent_bits}_m{self.mantissa_bits}{'fn' if self.finite_only else ''}{'uz' if self.unsigned_zero else ''}"
        elif self.kind == str:
            return f"str{self.bits}"
        elif self.kind == bool:
            return "bool"
        elif self.kind == object:
            return "object"
        else:
            return f"{self.kind.__name__}{self.bits}"

    @staticmethod
    def as_dtype(value: Union['DType', tuple, type, None]) -> Union['DType', None]:
        if isinstance(value, DType):
            return value
        elif value is int:
            return INT32
        elif value is float:
            from . import get_precision
            return DType.by_precision(float, get_precision())
        elif value is complex:
            from . import get_precision
            return DType.by_precision(complex, get_precision())
        elif value is None:
            return None
        elif isinstance(value, tuple):
            if len(value) == 2:
                return {
                    (int, 8): INT8,
                    (int, 16): INT16,
                    (int, 32): INT32,
                    (int, 64): INT64,
                    (float, 16): FLOAT16,
                    (float, 32): FLOAT32,
                    (float, 64): FLOAT64,
                    (complex, 64): COMPLEX64,
                    (complex, 128): COMPLEX128,
                }[value]
            return DType(*value)
        elif value is str:
            raise ValueError("str DTypes must specify bits")
        return {bool: BOOL, object: OBJECT}[value]

    @staticmethod
    def by_precision(kind: type, precision: int) -> 'DType':
        if kind == float:
            return {16: FLOAT16, 32: FLOAT32, 64: FLOAT64}[precision]
        elif kind == complex:
            return {32: COMPLEX64, 64: COMPLEX128}[precision]
        else:
            raise ValueError(f"Unsupported kind: {kind}")

    @staticmethod
    def int_by_bits(bits: int):
        return {8: INT8, 16: INT16, 32: INT32, 64: INT64}[bits]

    @staticmethod
    def by_bits(kind: type, bits: int):
        if kind is int:
            return {8: INT8, 16: INT16, 32: INT32, 64: INT64}[bits]
        elif kind is float:
            return {16: FLOAT16, 32: FLOAT32, 64: FLOAT64}[bits]
        elif kind is complex:
            return {64: COMPLEX64, 128: COMPLEX128}[bits]
        elif kind is str:
            return DType(str, bits, False, 0, 0, True, True)
        raise ValueError


# def get_dtype(kind: type, bits: int = None, precision: int = None):
#     assert kind in (bool, int, float, complex, str, object)
#     if kind is bool:
#         assert precision is None, f"Precision may only be specified for float or complex but got {kind}, precision={precision}"
#         return BOOL if bits is None else DType(bool, bits=bits, unsigned=True, exponent_bits=0, mantissa_bits=1, finite_only=True, unsigned_zero=True)
#     elif kind == object:
#         assert precision is None, f"Precision may only be specified for float or complex but got {kind}, precision={precision}"
#         return OBJECT if bits is None else DType(object, bits=bits, unsigned=True, exponent_bits=0, mantissa_bits=bits, finite_only=False, unsigned_zero=False)
#     elif precision is not None:
#         assert bits is None, "Specify either bits or precision when creating a DType but not both."
#         assert kind in [float, complex], f"Precision may only be specified for float or complex but got {kind}, precision={precision}"
#         if kind == float:
#             bits = precision
#         else:
#             bits = precision * 2
#     else:
#         assert isinstance(bits, int), f"bits must be an int but got {type(bits)}"


# --- NumPy Conversion ---

def to_numpy_dtype(dtype: DType):
    if dtype in _TO_NUMPY:
        return _TO_NUMPY[dtype]
    if dtype.kind == str:
        bytes_per_char = np.dtype('<U1').itemsize
        return np.dtype(f'<U{dtype.itemsize // bytes_per_char}')
    raise KeyError(f"Unsupported dtype: {dtype}")


def from_numpy_dtype(np_dtype) -> DType:
    if np_dtype in _FROM_NUMPY:
        return _FROM_NUMPY[np_dtype]
    else:
        for base_np_dtype, dtype in _FROM_NUMPY.items():
            if np_dtype == base_np_dtype:
                _FROM_NUMPY[np_dtype] = dtype
                return dtype
        if np_dtype.char == 'U':
            return DType(str, 8 * np_dtype.itemsize)
        raise ValueError(np_dtype)


BOOL = DType(bool, bits=8, unsigned=True, exponent_bits=0, mantissa_bits=1, finite_only=True, unsigned_zero=True)
OBJECT = DType(object, bits=int(np.round(np.log2(sys.maxsize))) + 1, unsigned=True, exponent_bits=0, mantissa_bits=int(np.round(np.log2(sys.maxsize))) + 1, finite_only=False, unsigned_zero=False)
# --- Int ---
INT8 = DType(int, 8, unsigned=False, exponent_bits=0, mantissa_bits=8, finite_only=True, unsigned_zero=True)
INT16 = DType(int, 16, unsigned=False, exponent_bits=0, mantissa_bits=16, finite_only=True, unsigned_zero=True)
INT32 = DType(int, 32, unsigned=False, exponent_bits=0, mantissa_bits=32, finite_only=True, unsigned_zero=True)
INT64 = DType(int, 64, unsigned=False, exponent_bits=0, mantissa_bits=64, finite_only=True, unsigned_zero=True)
UINT8 = DType(int, 8, unsigned=True, exponent_bits=0, mantissa_bits=8, finite_only=True, unsigned_zero=True)
UINT16 = DType(int, 16, unsigned=True, exponent_bits=0, mantissa_bits=16, finite_only=True, unsigned_zero=True)
UINT32 = DType(int, 32, unsigned=True, exponent_bits=0, mantissa_bits=32, finite_only=True, unsigned_zero=True)
UINT64 = DType(int, 64, unsigned=True, exponent_bits=0, mantissa_bits=64, finite_only=True, unsigned_zero=True)
# --- Float ---
BF16 = DType(float, 16, unsigned=False, exponent_bits=8, mantissa_bits=7, finite_only=False, unsigned_zero=False)
FLOAT16 = DType(float, 16, unsigned=False, exponent_bits=5, mantissa_bits=10, finite_only=False, unsigned_zero=False)
FLOAT32 = DType(float, 32, unsigned=False, exponent_bits=8, mantissa_bits=23, finite_only=False, unsigned_zero=False)
FLOAT64 = DType(float, 64, unsigned=False, exponent_bits=11, mantissa_bits=52, finite_only=False, unsigned_zero=False)
# --- Complex ---
COMPLEX64 = DType(complex, 64, unsigned=False, exponent_bits=8, mantissa_bits=23, finite_only=False, unsigned_zero=False)
COMPLEX128 = DType(complex, 128, unsigned=False, exponent_bits=11, mantissa_bits=52, finite_only=False, unsigned_zero=False)

_TO_NUMPY = {
    BOOL: np.bool_,
    OBJECT: object,
    # --- Int ---
    INT8: np.int8,
    INT16: np.int16,
    INT32: np.int32,
    INT64: np.int64,
    UINT8: np.uint8,
    UINT16: np.uint16,
    UINT32: np.uint32,
    UINT64: np.uint64,
    # --- Float ---
    FLOAT16: np.float16,
    FLOAT32: np.float32,
    FLOAT64: np.float64,
    # --- Complex ---
    COMPLEX64: np.complex64,
    COMPLEX128: np.complex128,
}
_FROM_NUMPY = {np: dtype for dtype, np in _TO_NUMPY.items()}
_FROM_NUMPY[np.bool_] = BOOL
_FROM_NUMPY[bool] = BOOL


UPCAST_LVL = {bool: 1, int: 2, float: 3, complex: 4, str: 5, object: 6}


@lru_cache
def combine_types(*dtypes: DType, fp_precision: int = None) -> DType:
    lvl = max(UPCAST_LVL[dt.kind] for dt in dtypes)
    if lvl == 1:  # all bool
        return dtypes[0]
    elif lvl == 2:  # all int / bool?
        return max(dtypes, key=lambda dt: dt.mantissa_bits)
    elif lvl == 3:  # all real?
        if isinstance(fp_precision, int):
            return DType.by_precision(float, fp_precision)
        else:
            highest_fp = max(dt.precision for dt in dtypes if dt.kind == float)
            return DType.by_precision(float, highest_fp)
    elif lvl == 4:  # complex
        if isinstance(fp_precision, int):
            return DType.by_precision(complex, fp_precision)
        else:
            highest_fp = max([dt.precision for dt in dtypes if dt.kind in (float, complex)])
            return DType.by_precision(complex, highest_fp)
    elif lvl == 5:  # string
        return max([dt for dt in dtypes if dt.kind == str], key=lambda dt: dt.bits)
    return OBJECT
