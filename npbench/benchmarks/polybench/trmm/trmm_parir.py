import parir
import torch

@parir.jit
def trmm(alpha, A, B, M, N):
    for i in range(M):
        parir.label('j')
        for j in range(N):
            parir.label('k')
            for k in range(i+1, M):
                B[i,j] += A[k,i] * B[k,j]
            B[i,j] = B[i,j] * alpha

def kernel(alpha, A, B):
    M, N = B.shape
    p = {
        'j': parir.threads(N),
        'k': parir.threads(256).reduce()
    }
    trmm(alpha, A, B, M, N, opts=parir.par(p))
