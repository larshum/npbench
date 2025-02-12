import parir
import torch

@parir.jit
def kernel_helper(path, N):
    for k in range(N):
        parir.label('i')
        for i in range(N):
            parir.label('j')
            for j in range(N):
                path[i, j] = min(path[i, j], path[i, k] + path[k, j])

def kernel(path):
    N, N = path.shape
    p = {
        'i': [parir.threads(N)],
        'j': [parir.threads(N)]
    }
    kernel_helper(path, N, parallelize=p)
