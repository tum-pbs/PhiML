from unittest import TestCase

from phiml import math
from phiml.math import spatial, wrap, squeeze, channel
from phiml.math.perm import all_permutations, optimal_perm

import torch


class TestTorchHooks(TestCase):

    def test_interpolate(self):
        t = math.randn(spatial(x=4, y=3))
        interpolated = torch.nn.functional.interpolate(t, (12, 9), mode='linear')
        self.assertEqual({'x','y'}, set(interpolated.shape.names))