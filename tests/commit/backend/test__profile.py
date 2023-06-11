from unittest import TestCase

from unifyml import math
from unifyml.backend import profile


class TestProfile(TestCase):

    def test_profile(self):
        with profile() as prof:
            math.ones() + math.ones()
        prof.print(min_duration=0)
