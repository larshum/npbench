import parpy

@parpy.jit
def parpy_kernel(L, x, b, N):
    parpy.label('N')
    for i in range(N):
        t = 0.0
        parpy.label('reduce')
        for k in range(i):
            t += L[i,k] * x[k]
        x[i] = (b[i] - t) / L[i,i]

def kernel(L, x, b):
    N = x.shape[0]
    p = {
        'N': parpy.threads(N),
        'reduce': parpy.threads(256).reduce()
    }
    parpy_kernel(L, x, b, N, opts=parpy.par(p))
