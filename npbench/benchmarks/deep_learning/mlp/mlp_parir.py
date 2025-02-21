import parir
import torch


@parir.jit
def relu_kernel(x, N, M):
    parir.label('i')
    for i in range(N):
        parir.label('j')
        for j in range(M):
            x[i,j] = max(x[i,j], 0.0)

def relu(x):
    N, M = x.shape
    p = { 'i': [parir.threads(N)], 'j': [parir.threads(M)] }
    relu_kernel(x, N, M, parallelize=p)
    return x

@parir.jit
def softmax_kernel(x, out, N, M):
    parir.label('i')
    for i in range(N):
        maxv = -parir.inf
        parir.label('jx')
        for j in range(M):
            maxv = max(maxv, x[i,j])
        parir.label('j')
        for j in range(M):
            out[i,j] = parir.exp(x[i,j] - maxv)
        s = parir.float32(0.0)
        parir.label('jx')
        for j in range(M):
            s += out[i,j]
        parir.label('j')
        for j in range(M):
            out[i,j] /= s

def softmax(x):
    N, M = x.shape
    out = torch.empty_like(x)
    p = {
        'i': [parir.threads(N)],
        'j': [parir.threads(1024)],
        'jx': [parir.threads(1024), parir.reduce()]
    }
    softmax_kernel(x, out, N, M, parallelize=p)
    return out

# 3-layer MLP
def mlp(input, w1, b1, w2, b2, w3, b3):
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    x = softmax(x @ w3 + b3)  # Softmax call can be omitted if necessary
    return x
