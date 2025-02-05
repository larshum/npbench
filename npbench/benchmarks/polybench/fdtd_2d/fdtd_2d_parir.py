import parir
from parir import ParKind
import torch

@parir.jit
def kernel_wrap(ex, ey, hz, _fict_, NX, NY, TMAX):
    for t in range(TMAX):
        for j in range(NY):
            ey[0, j] = _fict_[t]

        for i in range(1, NX):
            for j in range(NY):
                ey[i, j] = ey[i, j] - 0.5 * (hz[i, j] - hz[i-1, j])

        for i in range(NX):
            for j in range(1, NY):
                ex[i, j] = ex[i, j] - 0.5 * (hz[i, j] - hz[i, j-1])

        for i in range(0, NX-1):
            for j in range(0, NY-1):
                hz[i, j] = hz[i, j] - 0.7 * (ex[i, j+1] - ex[i, j] + ey[i+1, j] - ey[i, j])

def kernel(TMAX, ex, ey, hz, _fict_):
    NX, NY = ex.shape
    p = {
        'i': [ParKind.GpuThreads(NX-1)],
        'j': [ParKind.GpuThreads(1024)],
    }
    kernel_wrap(ex, ey, hz, _fict_, NX, NY, TMAX, parallelize=p)

