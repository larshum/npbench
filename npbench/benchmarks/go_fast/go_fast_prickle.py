# https://numba.readthedocs.io/en/stable/user/5minguide.html

import prickle
import torch

@prickle.jit
def prickle_kernel(a, tmp, out, N):
    prickle.label('i')
    for i in range(N):
        tmp[0] += prickle.tanh(a[i,i])
    prickle.label('ix')
    prickle.label('j')
    out[:,:] = a[:,:] + tmp[0]

def go_fast(a):
    N, N = a.shape
    tmp = torch.tensor([0.0], dtype=a.dtype, device=a.device)
    out = torch.empty_like(a)
    p = {
        'i': prickle.threads(1024).reduce(),
        'ix': prickle.threads(N),
        'j': prickle.threads(N)
    }
    prickle_kernel(a, tmp, out, N, opts=prickle.par(p))
    return out
