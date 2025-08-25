# https://numba.readthedocs.io/en/stable/user/5minguide.html

import prickle

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
    tmp = prickle.buffer.zeros((1,), a.dtype, a.backend)
    out = prickle.buffer.empty_like(a)
    p = {
        'i': prickle.threads(1024).reduce(),
        'ix': prickle.threads(N),
        'j': prickle.threads(N)
    }
    prickle_kernel(a, tmp, out, N, opts=prickle.par(p))
    return out
