from dataclasses import dataclass
from typing import Tuple, Sequence, Dict
from unittest import TestCase

from phiml.dataclasses import data_fields, config_fields, special_fields, sliceable, data_eq, equal
from phiml.math import Tensor, vec, wrap, shape, assert_close


class TestDataclasses(TestCase):

    def test_field_types(self):
        @dataclass
        class Custom:
            variable_attrs: Sequence[str]
            age: int
            next: 'Custom'
            conf: Dict[str, Sequence[Tuple[str, float, complex, bool, int, slice]]]
        data_names = [f.name for f in data_fields(Custom)]
        self.assertEqual(['next'], data_names)
        config_names = [f.name for f in config_fields(Custom)]
        self.assertEqual(['age', 'conf'], config_names)
        special_names = [f.name for f in special_fields(Custom)]
        self.assertEqual(['variable_attrs'], special_names)

    def test_sliceable(self):
        @sliceable
        @dataclass(frozen=True)
        class Custom:
            pos: Tensor
            edges: Dict[str, Tensor]
        c = Custom(vec(x=1, y=2), {'lo': wrap([-1, 1], 'b:b')})
        self.assertEqual(('b', 'vector'), shape(c).names)
        assert_close(c['y,x'].pos, c.pos['y,x'])
        assert_close(c.vector['y,x'].pos, c.pos['y,x'])

    def test_data_eq(self):
        @data_eq(abs_tolerance=.2)
        @dataclass(frozen=True)
        class Custom:
            pos: Tensor
        c1 = Custom(vec(x=0, y=1))
        c2 = Custom(vec(x=3, y=4))
        c11 = Custom(vec(x=.1, y=1.1))
        self.assertNotEqual(c1, c2)
        self.assertEqual(c1, c1)
        self.assertEqual(c1, c11)
        self.assertFalse(equal(c1, c11, abs_tolerance=0))
        self.assertTrue(equal(c1, c2, abs_tolerance=3))
