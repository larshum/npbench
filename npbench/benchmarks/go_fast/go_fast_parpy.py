# https://numba.readthedocs.io/en/stable/user/5minguide.html

import numpy as np
import parpy
from parpy.operators import tanh

@parpy.jit
def parpy_kernel(a, tmp, out, N):
    parpy.label('i')
    for i in range(N):
        tmp[0] += tanh(a[i,i])
    parpy.label('ix')
    parpy.label('j')
    out[:,:] = a[:,:] + tmp[0]

def go_fast(a):
    N, N = a.shape
    tmp = np.array([0.0], dtype=a.dtype.to_numpy())
    out = parpy.buffer.empty_like(a)
    p = {
        'i': parpy.threads(1024).reduce(),
        'ix': parpy.threads(N),
        'j': parpy.threads(N)
    }
    parpy_kernel(a, tmp, out, N, opts=parpy.par(p))
    return out
