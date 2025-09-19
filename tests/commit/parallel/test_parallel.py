from dataclasses import dataclass
from unittest import TestCase

from phiml import arange, batch
from phiml.parallel import parallel_property, parallel_compute
from phiml.math import Tensor, shape, math


@dataclass
class Data:
    value: Tensor

    @parallel_property
    def p1(self):
        return self.compute_p1(1)

    def compute_p1(self, add=1):
        return self.value + add

    @parallel_property(requires=shape)
    def p1_mean(self):
        return math.mean(self.p1, shape)

    @parallel_property
    def rel_p1(self):
        return self.p1 - self.p1_mean


class TestDataclasses(TestCase):

    def test_parallel_properties(self):
        data = Data(arange(batch(example=2)))
        req_properties = [Data.rel_p1]
        parallel_compute(data, req_properties, batch, max_workers=1, memory_limit=1, cache_dir="cache")
        assert not data.rel_p1.available
        math.assert_close([-.5, .5], data.rel_p1)
