import parir
import torch

@parir.jit
def softmax_wrap(x, out, N, H, SM):
    for i in range(N):
        for j in range(H):
            for k in range(SM):
                m = parir.float32(-parir.inf)
                for l in range(SM):
                    m = max(m, x[i,j,k,l])

                for l2 in range(SM):
                    out[i,j,k,l2] = parir.exp(x[i,j,k,l2]-m)

                s = parir.float32(0.0)
                for l in range(SM):
                    s = s + out[i,j,k,l]

                for l2 in range(SM):
                    out[i,j,k,l2] = out[i,j,k,l2] / s

# Numerically-stable version of softmax
def softmax(x):
    N, H, SM, SM = x.shape
    out = torch.zeros_like(x)
    p = {
        'i': [parir.threads(N)],
        'j': [parir.threads(H)],
        'k': [parir.threads(SM)],
        'l': [parir.threads(SM), parir.reduce()],
        'l2': [parir.threads(SM)]
    }
    softmax_wrap(x, out, N, H, SM, parallelize=p)
    return out
