from typing import Tuple

import numpy as np

from ._backend import Backend




def schedule_levels(b: Backend, matrix, lower_triangular=True):
    pass


def prepare_dense_triangular_solves(b: Backend, matrix, lower_triangular=True):
    pass


def bake_triangular_to_matmul(b: Backend, matrix, lower_triangular=True, preferred_matrix_volume=100):
    num_per_mul = 2


def bake_critical_points(b: Backend, matrix, lower_triangular=True):
    pass
