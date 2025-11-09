# https://numba.readthedocs.io/en/stable/user/5minguide.html

import torch


def go_fast(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += torch.tanh(a[i, i])
    return a + trace

go_fast_jit = torch.compile(go_fast)
