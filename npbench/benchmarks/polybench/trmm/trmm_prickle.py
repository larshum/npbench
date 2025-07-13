import prickle
import torch

@prickle.jit
def trmm(alpha, A, B, M, N):
    for i in range(M):
        prickle.label('j')
        for j in range(N):
            prickle.label('k')
            for k in range(i+1, M):
                B[i,j] += A[k,i] * B[k,j]
            B[i,j] = B[i,j] * alpha

def kernel(alpha, A, B):
    M, N = B.shape
    p = {
        'j': prickle.threads(N),
        'k': prickle.threads(256).reduce()
    }
    trmm(alpha, A, B, M, N, opts=prickle.par(p))
