import parir
import torch

@parir.jit
def kernel_wrap(ex, ey, hz, _fict_, NX, NY, TMAX):
    for t in range(TMAX):
        parir.label('j')
        for j in range(NY):
            ey[0, j] = _fict_[t]

        parir.label('i')
        for i in range(1, NX):
            parir.label('j')
            for j in range(NY):
                ey[i, j] -= 0.5 * (hz[i, j] - hz[i-1, j])

        parir.label('i')
        for i in range(NX):
            parir.label('j')
            for j in range(1, NY):
                ex[i, j] -= 0.5 * (hz[i, j] - hz[i, j-1])

        parir.label('i')
        for i in range(0, NX-1):
            parir.label('j')
            for j in range(0, NY-1):
                hz[i, j] -= 0.7 * (ex[i, j+1] - ex[i, j] + ey[i+1, j] - ey[i, j])

def kernel(TMAX, ex, ey, hz, _fict_):
    NX, NY = ex.shape
    p = {
        'i': [parir.threads(NX-1)],
        'j': [parir.threads(1024)],
    }
    kernel_wrap(ex, ey, hz, _fict_, NX, NY, TMAX, parallelize=p)

