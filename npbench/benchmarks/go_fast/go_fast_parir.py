# https://numba.readthedocs.io/en/stable/user/5minguide.html

import parir
from parir import ParKind
import torch

@parir.jit
def parir_kernel(a, tmp, out, N):
    for i in range(N):
        tmp[0] = tmp[0] + parir.tanh(a[i, i])
    for ix in range(N):
        for j in range(N):
            out[ix, j] = a[ix, j] + tmp[0]

def go_fast(a):
    N, N = a.shape
    tmp = torch.tensor([0.0], dtype=a.dtype, device=a.device)
    out = torch.empty_like(a)
    p = {
        'i': [ParKind.GpuThreads(1024), ParKind.GpuReduction()],
        'ix': [ParKind.GpuThreads(N)],
        'j': [ParKind.GpuThreads(N)]
    }
    parir_kernel(a, tmp, out, N, parallelize=p)
    return out
