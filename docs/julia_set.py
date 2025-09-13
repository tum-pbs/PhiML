from dataclasses import dataclass
from functools import cached_property

from phiml import Tensor, iterate


def belongs_to_julia_set(z, c, iter_count: int):
    return iterate(lambda z, k: (z ** 2 + c, k + (abs(z) < 2)), iter_count, z, 0)[1]


@dataclass
class JuliaSet:
    c: Tensor
    z0: Tensor
    iter_count: int = 50

    @cached_property
    def values(self):
        return belongs_to_julia_set(self.z0, self.c, self.iter_count)
