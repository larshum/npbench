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
                parir.label('l')
                m = parir.max(x[i,j,k,:])

                parir.label('l')
                out[i,j,k,:] = parir.exp(x[i,j,k,:]-m)

                parir.label('l')
                s = parir.sum(out[i,j,k,:])

                parir.label('l')
                out[i,j,k,:] /= s

# Numerically-stable version of softmax
def softmax(x):
    N, H, SM, SM = x.shape
    out = torch.zeros_like(x)
    p = {
        'i': [parir.threads(N)],
        'j': [parir.threads(H)],
        'k': [parir.threads(SM)],
        'l': [parir.threads(256)],
    }
    softmax_wrap(x, out, N, H, SM, parallelize=p)
    return out
