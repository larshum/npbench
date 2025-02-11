import parir
from parir import ParKind
import torch


@parir.jit
def relu_kernel(x, N, M):
    for i in range(N):
        for j in range(M):
            x[i,j] = max(x[i,j], 0.0)

def relu(x):
    N, M = x.shape
    p = { 'i': [ParKind.GpuThreads(N)], 'j': [ParKind.GpuThreads(M)] }
    relu_kernel(x, N, M, parallelize=p)
    return x

@parir.jit
def softmax_kernel(x, out, N, M):
    for i in range(N):
        maxv = -parir.inf
        for jx in range(M):
            maxv = max(maxv, x[i,jx])
        for j in range(M):
            out[i,j] = parir.exp(x[i,j] - maxv)
        s = parir.float32(0.0)
        for jx in range(M):
            s = s + out[i,jx]
        for j in range(M):
            out[i,j] = out[i,j] / s

def softmax(x):
    N, M = x.shape
    out = torch.empty_like(x)
    p = {
        'i': [ParKind.GpuThreads(N)],
        'j': [ParKind.GpuThreads(1024)],
        'jx': [ParKind.GpuThreads(1024), ParKind.GpuReduction()]
    }
    softmax_kernel(x, out, N, M, parallelize=p)
    return out

# 3-layer MLP
def mlp(input, w1, b1, w2, b2, w3, b3):
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    x = softmax(x @ w3 + b3)  # Softmax call can be omitted if necessary
    return x
