import parir
from parir import ParKind
import torch

@parir.jit
def kernel_step(A, B, N):
    for i in range(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):
                B[i, j, k] = (0.125 * (A[i+1, j, k] - 2.0 * A[i, j, k] + A[i-1, j, k]) +
                              0.125 * (A[i, j+1, k] - 2.0 * A[i, j, k] + A[i, j-1, k]) +
                              0.125 * (A[i, j, k+1] - 2.0 * A[i, j, k] + A[i, j, k-1]) +
                              A[i, j, k])
    for i in range(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):
                A[i, j, k] = (0.125 * (B[i+1, j, k] - 2.0 * B[i, j, k] + B[i-1, j, k]) +
                              0.125 * (B[i, j+1, k] - 2.0 * B[i, j, k] + B[i, j-1, k]) +
                              0.125 * (B[i, j, k+1] - 2.0 * B[i, j, k] + B[i, j, k-1]) +
                              B[i, j, k])

def kernel(TSTEPS, A, B):
    N, N, N = A.shape
    p = {
        'i': [ParKind.GpuThreads(N-2)],
        'j': [ParKind.GpuThreads(N-2)],
        'k': [ParKind.GpuThreads(32)]
    }
    for t in range(1, TSTEPS):
        kernel_step(A, B, N, parallelize=p)
