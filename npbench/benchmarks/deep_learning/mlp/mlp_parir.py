import parir
import torch


def relu(x):
    return torch.maximum(x, torch.empty_like(x))

@parir.jit
def softmax_kernel(x, out, N, M):
    parir.label('i')
    for i in range(N):
        parir.label('j')
        maxv = parir.max(x[i,:])
        parir.label('j')
        out[i,:] = parir.exp(x[i,:] - maxv)
        parir.label('j')
        s = parir.sum(out[i,:])
        parir.label('j')
        out[i,:] /= s

def softmax(x):
    N, M = x.shape
    out = torch.empty_like(x)
    p = {
        'i': [parir.threads(N)],
        'j': [parir.threads(1024)],
    }
    softmax_kernel(x, out, N, M, parallelize=p)
    return out

# 3-layer MLP
def mlp(input, w1, b1, w2, b2, w3, b3):
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    x = softmax(x @ w3 + b3)  # Softmax call can be omitted if necessary
    return x
