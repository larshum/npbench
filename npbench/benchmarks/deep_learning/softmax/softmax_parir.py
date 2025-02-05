import parir
from parir import ParKind
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
        'i': [ParKind.GpuThreads(N)],
        'j': [ParKind.GpuThreads(H)],
        'k': [ParKind.GpuThreads(SM)],
        'l': [ParKind.GpuThreads(SM), ParKind.GpuReduction()],
        'l2': [ParKind.GpuThreads(SM)]
    }
    softmax_wrap(x, out, N, H, SM, parallelize=p)
    return out
