import parir
from parir import ParKind
import torch

@parir.jit
def trmm(alpha, A, B, M, N):
    for i in range(M):
        for j in range(N):
            for k in range(i+1, M):
                B[i,j] = B[i,j] + (A[k,i] * B[k,j])
            B[i,j] = B[i,j] * alpha

def kernel(alpha, A, B):
    M, N = B.shape
    p = {
        'j': [ParKind.GpuThreads(N)],
        'k': [ParKind.GpuThreads(256), ParKind.GpuReduction()]
    }
    trmm(alpha, A, B, M, N, parallelize=p)
