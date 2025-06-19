import parir
import torch


@parir.jit
def parir_kernel(alpha, beta, C, A, B, M, N):
    parir.label('M')
    parir.label('N')
    C[:,:] *= beta

    for i in range(M):
        parir.label('N')
        for j in range(N):
            parir.label('M')
            for k in range(i):
                C[k,j] += alpha * B[i,j] * A[i,k]

        parir.label('N')
        for j in range(N):
            dot_sum = 0.0
            parir.label('i_red')
            for k in range(i):
                dot_sum += B[k,j] * A[i,k]
            C[i,j] += alpha * B[i,j] * A[i,i] + alpha * dot_sum

def kernel(alpha, beta, C, A, B):
    M, N = C.shape
    p = {
        'M': parir.threads(M),
        'N': parir.threads(N),
        'i_red': parir.threads(32).reduce()
    }
    parir_kernel(alpha, beta, C, A, B, M, N, opts=parir.parallelize(p))
