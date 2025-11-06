import parpy
from parpy.math import exp

@parpy.jit
def softmax_wrap(x, out, N, H, SM):
    parpy.label('i')
    for i in range(N):
        parpy.label('j')
        for j in range(H):
            parpy.label('k')
            for k in range(SM):
                parpy.label('l')
                m = parpy.reduce.max(x[i,j,k,:])

                parpy.label('l')
                out[i,j,k,:] = exp(x[i,j,k,:]-m)

                parpy.label('l')
                s = parpy.reduce.sum(out[i,j,k,:])

                parpy.label('l')
                out[i,j,k,:] /= s

# Numerically-stable version of softmax
def softmax(x):
    N, H, SM, SM = x.shape
    out = parpy.buffer.zeros_like(x)
    p = {
        'i': parpy.threads(N),
        'j': parpy.threads(H),
        'k': parpy.threads(SM),
        'l': parpy.threads(256),
    }
    softmax_wrap(x, out, N, H, SM, opts=parpy.par(p))
    return out
