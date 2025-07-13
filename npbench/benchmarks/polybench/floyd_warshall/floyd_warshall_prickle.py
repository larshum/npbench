import prickle
import torch

@prickle.jit
def kernel_helper(path, N):
    for k in range(N):
        prickle.label('i')
        for i in range(N):
            prickle.label('j')
            path[i,:] = prickle.min(path[i,:], path[i,k] + path[k,:])

def kernel(path):
    N, N = path.shape
    p = {
        'i': prickle.threads(N),
        'j': prickle.threads(N)
    }
    kernel_helper(path, N, opts=prickle.par(p))
