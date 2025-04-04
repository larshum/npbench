import parir
import torch

@parir.jit
def kernel_wrap(ex, ey, hz, _fict_, TMAX):
    for t in range(TMAX):
        parir.label('j')
        ey[0, :] = _fict_[t]

        parir.label('i')
        parir.label('j')
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])

        parir.label('i')
        parir.label('j')
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])

        parir.label('i')
        parir.label('j')
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])

def kernel(TMAX, ex, ey, hz, _fict_):
    NX, NY = ex.shape
    p = {
        'i': parir.threads(NX-1),
        'j': parir.threads(1024),
    }
    kernel_wrap(ex, ey, hz, _fict_, TMAX, parallelize=p)

