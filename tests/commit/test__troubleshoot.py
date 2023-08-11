from unittest import TestCase

from phiml._troubleshoot import assert_minimal_config, troubleshoot, count_tensors_in_memory, plot_solves
from phiml import math

class TestTroubleshoot(TestCase):

    def test_assert_minimal_config(self):
        assert_minimal_config()

    def test_troubleshoot(self):
        troubleshoot()

    def test_count_tensors_in_memory(self):
        count_tensors_in_memory()

    def test_plot_solves(self):
        with plot_solves():
            math.solve_linear(lambda x: 2 * x, math.tensor(1.), math.Solve(x0=0))
