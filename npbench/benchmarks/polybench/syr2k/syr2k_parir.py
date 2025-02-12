import parir
import torch

@parir.jit
def kernel_wrap(alpha, beta, C, A, B, N, M):
    parir.label('i')
    for i in range(N):
        for j in range(i+1):
            C[i,j] *= beta
            parir.label('k')
            for k in range(M):
                C[i,j] += A[j,k] * alpha * B[i,k] + B[j,k] * alpha * A[i,k]

def kernel(alpha, beta, C, A, B):
    N, M = A.shape
    p = {
        'i': [parir.threads(N)],
        'k': [parir.threads(256), parir.reduce()]
    }
    kernel_wrap(alpha, beta, C, A, B, N, M, parallelize=p)

