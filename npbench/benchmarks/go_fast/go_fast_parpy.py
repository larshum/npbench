# https://numba.readthedocs.io/en/stable/user/5minguide.html

import numpy as np
import parpy
from parpy.math import tanh

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
        'i': parpy.threads(1024).par_reduction(),
        'ix': parpy.threads(N),
        'j': parpy.threads(N)
    }
    opts = parpy.par(p)
    opts.max_unroll_count = 0
    parpy_kernel(a, tmp, out, N, opts=opts)
    return out
