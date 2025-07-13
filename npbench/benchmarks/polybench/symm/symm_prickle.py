import prickle
import torch


@prickle.jit
def prickle_kernel(alpha, beta, C, A, B, M, N):
    prickle.label('M')
    prickle.label('N')
    C[:,:] *= beta

    for i in range(M):
        prickle.label('N')
        for j in range(N):
            prickle.label('M')
            for k in range(i):
                C[k,j] += alpha * B[i,j] * A[i,k]

        prickle.label('N')
        for j in range(N):
            dot_sum = 0.0
            prickle.label('i_red')
            for k in range(i):
                dot_sum += B[k,j] * A[i,k]
            C[i,j] += alpha * B[i,j] * A[i,i] + alpha * dot_sum

def kernel(alpha, beta, C, A, B):
    M, N = C.shape
    p = {
        'M': prickle.threads(M),
        'N': prickle.threads(N),
        'i_red': prickle.threads(32).reduce()
    }
    prickle_kernel(alpha, beta, C, A, B, M, N, opts=prickle.par(p))
