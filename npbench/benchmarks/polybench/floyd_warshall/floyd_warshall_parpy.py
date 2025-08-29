import parpy
from parpy.operators import min
import torch

@parpy.jit
def kernel_helper(path, N):
    for k in range(N):
        parpy.label('i')
        for i in range(N):
            parpy.label('j')
            path[i,:] = min(path[i,:], path[i,k] + path[k,:])

def kernel(path):
    N, N = path.shape
    p = {
        'i': parpy.threads(N),
        'j': parpy.threads(N)
    }
    kernel_helper(path, N, opts=parpy.par(p))
