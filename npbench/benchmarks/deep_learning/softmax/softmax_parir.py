import parir
import torch

@parir.jit
def softmax_wrap(x, out, N, H, SM):
    parir.label('i')
    for i in range(N):
        parir.label('j')
        for j in range(H):
            parir.label('k')
            for k in range(SM):
                m = parir.float32(-parir.inf)
                parir.label('l')
                for l in range(SM):
                    m = max(m, x[i,j,k,l])

                parir.label('l2')
                for l in range(SM):
                    out[i,j,k,l] = parir.exp(x[i,j,k,l]-m)

                s = parir.float32(0.0)
                parir.label('l')
                for l in range(SM):
                    s += out[i,j,k,l]

                parir.label('l2')
                for l in range(SM):
                    out[i,j,k,l] /= s

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
