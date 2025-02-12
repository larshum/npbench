import parir
import torch

@parir.jit
def parir_kernel(A, B, N, TSTEPS):
    for t in range(1, TSTEPS):
        parir.label('i')
        for i in range(1, N-1):
            parir.label('j')
            for j in range(1, N-1):
                parir.label('k')
                for k in range(1, N-1):
                    B[i, j, k] = (0.125 * (A[i+1, j, k] - 2.0 * A[i, j, k] + A[i-1, j, k]) +
                                  0.125 * (A[i, j+1, k] - 2.0 * A[i, j, k] + A[i, j-1, k]) +
                                  0.125 * (A[i, j, k+1] - 2.0 * A[i, j, k] + A[i, j, k-1]) +
                                  A[i, j, k])
        parir.label('i')
        for i in range(1, N-1):
            parir.label('j')
            for j in range(1, N-1):
                parir.label('k')
                for k in range(1, N-1):
                    A[i, j, k] = (0.125 * (B[i+1, j, k] - 2.0 * B[i, j, k] + B[i-1, j, k]) +
                                  0.125 * (B[i, j+1, k] - 2.0 * B[i, j, k] + B[i, j-1, k]) +
                                  0.125 * (B[i, j, k+1] - 2.0 * B[i, j, k] + B[i, j, k-1]) +
                                  B[i, j, k])

def kernel(TSTEPS, A, B):
    N, N, N = A.shape
    p = {
        'i': [parir.threads(64)],
        'j': [parir.threads(64)],
        'k': [parir.threads(64)]
    }
    parir_kernel(A, B, N, TSTEPS, parallelize=p)
