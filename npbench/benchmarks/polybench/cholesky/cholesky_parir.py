import parir
import torch


@parir.jit
def parir_kernel(A, N):
    with parir.gpu:
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
    p = { 'k': parir.threads(256).reduce() }
    parir_kernel(A, N, parallelize=p)
