import parir
import torch

@parir.jit
def kernel_wrap(A, B, N, TSTEPS):
    for t in range(TSTEPS):
        for i in range(1, N-1):
            B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        for i in range(1, N-1):
            A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])

def kernel(TSTEPS, A, B):
    N, = A.shape
    p = {'i': [parir.threads(N-2)]}
    kernel_wrap(A, B, N, TSTEPS, parallelize=p)
