import numpy as np
import prickle

def relu(x):
    return np.maximum(x, 0)

@prickle.jit
def softmax_kernel(x, out, N, M):
    prickle.label('i')
    for i in range(N):
        prickle.label('j')
        maxv = prickle.max(x[i,:])
        prickle.label('j')
        out[i,:] = prickle.exp(x[i,:] - maxv)
        prickle.label('j')
        s = prickle.sum(out[i,:])
        prickle.label('j')
        out[i,:] /= s

def softmax(x, backend):
    N, M = x.shape
    x = prickle.buffer.Buffer.from_array(x, backend)
    out = prickle.buffer.empty_like(x)
    p = {
        'i': prickle.threads(N),
        'j': prickle.threads(1024),
    }
    softmax_kernel(x, out, N, M, opts=prickle.par(p))
    return out

# 3-layer MLP
def mlp(input, w1, b1, w2, b2, w3, b3):
    backend = input.backend
    x = relu(input.numpy() @ w1.numpy() + b1.numpy())
    x = relu(x @ w2.numpy() + b2.numpy())
    x = softmax(x @ w3.numpy() + b3.numpy(), backend)  # Softmax call can be omitted if necessary
    return x
