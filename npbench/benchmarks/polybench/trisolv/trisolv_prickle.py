import prickle

@prickle.jit
def prickle_kernel(L, x, b, N):
    prickle.label('N')
    for i in range(N):
        t = 0.0
        prickle.label('reduce')
        for k in range(i):
            t += L[i,k] * x[k]
        x[i] = (b[i] - t) / L[i,i]

def kernel(L, x, b):
    N = x.shape[0]
    p = {
        'N': prickle.threads(N),
        'reduce': prickle.threads(256).reduce()
    }
    prickle_kernel(L, x, b, N, opts=prickle.par(p))
