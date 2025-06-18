import parir
import torch

@parir.jit
def kernel_helper(path, N):
    for k in range(N):
        parir.label('i')
        for i in range(N):
            parir.label('j')
            path[i,:] = parir.min(path[i,:], path[i,k] + path[k,:])

def kernel(path):
    N, N = path.shape
    p = {
        'i': parir.threads(N),
        'j': parir.threads(N)
    }
    kernel_helper(path, N, opts=parir.parallelize(p))
