import parir
import torch

@parir.jit
def kernel_wrap(A, B, N, TSTEPS):
    for t in range(1, TSTEPS):
        for i in range(1, N-1):
            for j in range(1, N-1):
                B[i, j] = 0.2 * (A[i,j] + A[i,j-1] + A[i,j+1] + A[i+1,j] + A[i-1,j])
        for i in range(1, N-1):
            for j in range(1, N-1):
                A[i, j] = 0.2 * (B[i,j] + B[i,j-1] + B[i,j+1] + B[i+1,j] + B[i-1,j])


def kernel(TSTEPS, A, B):
    N, N = A.shape
    p = {
        'i': [parir.threads(N-1)],
        'j': [parir.threads(N-1)],
    }
    kernel_wrap(A, B, N, TSTEPS, parallelize=p)
