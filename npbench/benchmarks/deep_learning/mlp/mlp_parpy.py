import numpy as np
import parpy
from parpy.math import exp
import torch


def relu(x):
    return torch.maximum(x, torch.zeros_like(x))

def relu_numpy(x):
    return np.maximum(x, 0)

@parpy.jit
def softmax_kernel(x, out, N, M):
    parpy.label('i')
    for i in range(N):
        parpy.label('j')
        maxv = parpy.reduce.max(x[i,:])
        parpy.label('j')
        out[i,:] = exp(x[i,:] - maxv)
        parpy.label('j')
        s = parpy.reduce.sum(out[i,:])
        parpy.label('j')
        out[i,:] /= s

def softmax(x, backend):
    x = parpy.buffer.from_array(x, backend)
    N, M = x.shape
    out = parpy.buffer.empty_like(x)
    p = {
        'i': parpy.threads(N),
        'j': parpy.threads(1024),
    }
    softmax_kernel(x, out, N, M, opts=parpy.par(p))
    return out

# 3-layer MLP
def mlp(input, w1, b1, w2, b2, w3, b3):
    # Use PyTorch for CUDA and NumPy for Metal to minimize overhead
    if input.backend() == parpy.CompileBackend.Cuda:
        x = relu(input.torch() @ w1.torch() + b1.torch())
        x = relu(x @ w2.torch() + b2.torch())
        return softmax(x @ w3.torch() + b3.torch(), parpy.CompileBackend.Cuda)
    elif input.backend() == parpy.CompileBackend.Metal:
        x = relu_numpy(np.asarray(input) @ np.asarray(w1) + np.asarray(b1))
        x = relu_numpy(x @ np.asarray(w2) + np.asarray(b2))
        return softmax(x @ np.asarray(w3) + np.asarray(b3), parpy.CompileBackend.Metal)
    else:
        print(f"Unsupported backend: {input.backend()}")
