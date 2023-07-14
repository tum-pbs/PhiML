from unittest import TestCase

from ml4s import math
from ml4s.backend import profile


class TestProfile(TestCase):

    def test_profile(self):
        with profile() as prof:
            math.ones() + math.ones()
        prof.print(min_duration=0)
