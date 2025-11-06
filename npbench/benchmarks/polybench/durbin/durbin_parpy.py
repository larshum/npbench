import parpy

@parpy.jit
def parpy_kernel(r, y, temp, N):
    with parpy.gpu:
        alpha = -r[0]
        beta = 1.0
        y[0] = -r[0]
        for k in range(1, N):
            beta *= 1.0 - alpha * alpha
            t = 0.0
            parpy.label('k_red')
            for i in range(k):
                t += r[k-i-1] * y[i]
            alpha = -(r[k] + t) / beta
            parpy.label('k')
            for i in range(k):
                temp[i] = alpha * y[k-i-1]
            parpy.label('k')
            for i in range(k):
                y[i] += temp[i]
            y[k] = alpha

def kernel(r):
    y = parpy.buffer.empty_like(r)
    temp = parpy.buffer.empty_like(y)
    N, = r.shape
    p = {
        'k_red': parpy.threads(512).par_reduction(),
        'k': parpy.threads(512)
    }
    parpy_kernel(r, y, temp, N, opts=parpy.par(p))
    return y
