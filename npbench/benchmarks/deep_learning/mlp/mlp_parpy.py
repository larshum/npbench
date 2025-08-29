import parpy
from parpy.operators import exp, max, sum
import torch


def relu(x):
    return torch.maximum(x, torch.zeros_like(x))

@parpy.jit
def softmax_kernel(x, out, N, M):
    parpy.label('i')
    for i in range(N):
        parpy.label('j')
        maxv = max(x[i,:])
        parpy.label('j')
        out[i,:] = exp(x[i,:] - maxv)
        parpy.label('j')
        s = sum(out[i,:])
        parpy.label('j')
        out[i,:] /= s

def softmax(x):
    N, M = x.shape
    out = torch.empty_like(x)
    p = {
        'i': parpy.threads(N),
        'j': parpy.threads(1024),
    }
    softmax_kernel(x, out, N, M, opts=parpy.par(p))
    return out

# 3-layer MLP
def mlp(input, w1, b1, w2, b2, w3, b3):
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    x = softmax(x @ w3 + b3)  # Softmax call can be omitted if necessary
    return x
