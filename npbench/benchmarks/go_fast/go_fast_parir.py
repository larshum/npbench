# https://numba.readthedocs.io/en/stable/user/5minguide.html

import parir
import torch

@parir.jit
def parir_kernel(a, tmp, out, N):
    parir.label('i')
    for i in range(N):
        tmp[0] += parir.tanh(a[i,i])
    parir.label('ix')
    parir.label('j')
    out[:,:] = a[:,:] + tmp[0]

def go_fast(a):
    N, N = a.shape
    tmp = torch.tensor([0.0], dtype=a.dtype, device=a.device)
    out = torch.empty_like(a)
    p = {
        'i': parir.threads(1024).reduce(),
        'ix': parir.threads(N),
        'j': parir.threads(N)
    }
    parir_kernel(a, tmp, out, N, parallelize=p)
    return out
