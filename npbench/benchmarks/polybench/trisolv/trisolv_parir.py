import parir
import torch


@parir.jit
def parir_kernel(L, x, b, N):
    parir.label('N')
    for i in range(N):
        t = 0.0
        parir.label('reduce')
        for k in range(i):
            t += L[i,k] * x[k]
        x[i] = (b[i] - t) / L[i,i]

def kernel(L, x, b):
    N = x.shape[0]
    p = {
        'N': parir.threads(N),
        'reduce': parir.threads(256).reduce()
    }
    parir_kernel(L, x, b, N, parallelize=p)
