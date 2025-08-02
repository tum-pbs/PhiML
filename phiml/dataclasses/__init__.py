"""
PhiML makes it easy to work with custom classes.
Any class decorated with `@dataclass` can be used with `phiml.math` functions, such as `shape()`, `slice()`, `stack()`, `concat()`, `expand()` and many more.
We recommend always setting `frozen=True`.

PhiML's dataclass support explicitly handles properties defined decorated with `functools.cached_property`.
Their cached values will be preserved whenever possible, preventing costly re-computation when modifying unrelated properties, slicing, gathering, or stacking objects.
This will usually affect all *data fields*, i.e. fields that hold `Tensor` or composite properties.

Dataclass fields can additionally be specified as being *variable* and *value*.
This affects which data_fields are optimized / traced by functions like `phiml.math.jit_compile` or `phiml.math.minimize`.

**Template for custom classes:**

>>> from typing import Tuple from dataclasses import dataclass
>>> from phiml.dataclasses import sliceable, cached_property
>>> from phiml.math import Tensor, Shape, shape
>>>
>>> @sliceable
>>> @dataclass(frozen=True)
>>> class MyClass:
>>>     # --- Attributes ---
>>>     attribute1: Tensor
>>>     attribute2: 'MyClass' = None
>>>
>>>     # --- Additional fields ---
>>>     field1: str = 'x'
>>>
>>>     # --- Special fields declaring attribute types. Must be of type Tuple[str, ...] ---
>>>     variable_attrs: Tuple[str, ...] = ('attribute1', 'attribute2')
>>>     value_attrs: Tuple[str, ...] = ()
>>>
>>>     def __post_init__(self):
>>>         assert self.field1 in 'xyz'
>>>
>>>     @cached_property
>>>     def shape(self) -> Shape:  # override the default shape which is merged from all attribute shapes
>>>         return self.attribute1.shape & shape(self.attribute2)
>>>
>>>     @cached_property  # the cache will be copied to derived instances unless attribute1 changes (this is analyzed from the code)
>>>     def derived_property(self) -> Tensor:
>>>         return self.attribute1 + 1
"""

from functools import cached_property

from ._dataclasses import sliceable, data_fields, non_data_fields, config_fields, special_fields, replace, copy, getitem, equal, data_eq

from ._parallel import parallel_compute, parallel_property
from ._tensor_cache import on_load_into_memory, get_cache_files, set_cache_ttl

__all__ = [key for key in globals().keys() if not key.startswith('_')]
