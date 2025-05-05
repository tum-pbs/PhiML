from dataclasses import dataclass
from typing import Tuple, Sequence, Dict
from unittest import TestCase

from phiml import os


class TestOS(TestCase):

    def test_list_files(self):
        os.list_files(os.path.dirname(os.path.abspath(".")))

    def test_is_dir(self):
        dir = os.path.dirname(os.path.abspath("."))
        self.assertTrue(os.path.isdir(dir))
