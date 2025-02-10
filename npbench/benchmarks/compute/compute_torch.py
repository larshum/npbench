# https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html

import torch


def compute(array_1, array_2, a, b, c):
    return torch.clip(array_1, 2, 10) * a + array_2 * b + c
