import parir
import torch


@parir.jit
def parir_kernel(A, N):
    # We add this loop so that we can perform operations outside any "actual"
    # parallelism.
    parir.label('outer')
    for x in range(1):
        A[0,0] = parir.sqrt(A[0,0])
        for i in range(1, N):
            for j in range(i):
                s = 0.0
                parir.label('k')
                for k in range(j):
                    s += A[i,k] * A[j,k]
                A[i,j] -= s
                A[i,j] = A[i,j] / A[j,j]
            s = 0.0
            for k in range(i):
                s += A[i,k] * A[i,k]
            A[i,i] -= s
            A[i,i] = parir.sqrt(A[i,i])

def kernel(A):
    N, _ = A.shape
    p = {
        'outer': [parir.threads(2)],
        'k': [parir.threads(256), parir.reduce()]
    }
    parir_kernel(A, N, parallelize=p)
    return

    A[0, 0] = torch.sqrt(A[0, 0])
    for i in range(1, A.shape[0]):
        for j in range(i):
            A[i, j] -= torch.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= torch.dot(A[i, :i], A[i, :i])
        A[i, i] = torch.sqrt(A[i, i])
