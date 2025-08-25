import prickle

@prickle.jit
def softmax_wrap(x, out, N, H, SM):
    prickle.label('i')
    for i in range(N):
        prickle.label('j')
        for j in range(H):
            prickle.label('k')
            for k in range(SM):
                prickle.label('l')
                m = prickle.max(x[i,j,k,:])

                prickle.label('l')
                out[i,j,k,:] = prickle.exp(x[i,j,k,:]-m)

                prickle.label('l')
                s = prickle.sum(out[i,j,k,:])

                prickle.label('l')
                out[i,j,k,:] /= s

# Numerically-stable version of softmax
def softmax(x):
    N, H, SM, SM = x.shape
    out = prickle.buffer.zeros_like(x)
    p = {
        'i': prickle.threads(N),
        'j': prickle.threads(H),
        'k': prickle.threads(SM),
        'l': prickle.threads(256),
    }
    softmax_wrap(x, out, N, H, SM, opts=prickle.par(p))
    return out
