import parir
from parir import ParKind
import torch

@parir.jit
def kernel_wrap(alpha, beta, C, A, B, N, M):
    for i in range(N):
        for j in range(i+1):
            C[i,j] = C[i,j] * beta
            for k in range(M):
                C[i,j] = C[i,j] + (A[j,k] * alpha * B[i,k] + B[j,k] * alpha * A[i,k])

def kernel(alpha, beta, C, A, B):
    N, M = A.shape
    p = {
        'i': [ParKind.GpuThreads(N)],
        'k': [ParKind.GpuThreads(256), ParKind.GpuReduction()]
    }
    kernel_wrap(alpha, beta, C, A, B, N, M, parallelize=p)

