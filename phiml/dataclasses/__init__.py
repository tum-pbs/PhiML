"""
PhiML makes it easy to work with custom classes.
Any class decorated with `@dataclass` can be used with `phiml.math` functions, such as `shape()`, `slice()`, `stack()`, `concat()`, `expand()` and many more.
We recommend always setting `frozen=True`.

PhiML's dataclass support explicitly handles properties defined decorated with `functools.cached_property`.
Their cached values will be preserved whenever possible, preventing costly re-computation when modifying unrelated properties, slicing, gathering, or stacking objects.
This will usually affect all *attributes*, i.e. fields that hold `Tensor` or composite properties.

Dataclass fields can additionally be specified as being *variable* and *value*.
This affects which attributes are optimized / traced by functions like `phiml.math.jit_compile` or `phiml.math.minimize`.

**Template for custom classes**
```python
from functools import cached_property
from phiml.dataclasses import dataclass, getitem
from phiml.math import Tensor, shape, exp

@dataclass(frozen=True)
class MyClass:
    attribute1: Tensor
    attribute2: 'MyClass' = None

    field1: str = 'x'

    variable_attrs = ('attribute1', 'attribute2')
    value_attrs = ()

    def __post_init__(self):
        assert self.field1 in 'xyz'

    @cached_property
    def shape(self):
        return self.attribute1.shape & shape(self.attribute2)

    @cached_property  # the cache will be copied to derived instances unless attribute1 changes (this is analyzed from the code)
    def derived_property(self):
        return exp(self.attribute1)

    def __getitem__(self, item):
        return getitem(self, item)
```
"""

from dataclasses import dataclass

from ._dataclasses import attributes, replace, getitem

__all__ = [key for key in globals().keys() if not key.startswith('_')]
