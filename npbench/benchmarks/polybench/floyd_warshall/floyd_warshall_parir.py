import parir
from parir import ParKind
import torch

@parir.jit
def kernel_helper(path, N):
    for k in range(N):
        for i in range(N):
            for j in range(N):
                path[i, j] = min(path[i, j], path[i, k] + path[k, j])

def kernel(path):
    N, N = path.shape
    p = {
        'i': [ParKind.GpuThreads(N)],
        'j': [ParKind.GpuThreads(N)]
    }
    kernel_helper(path, N, parallelize=p)
