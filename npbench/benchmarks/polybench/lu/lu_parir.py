import parir
import torch


@parir.jit
def lu_parir(A, N):
    with parir.gpu:
        for i in range(N):
            for j in range(i):
                s = 0.0
                parir.label('k')
                for k in range(j):
                    s += A[i,k] * A[k,j]
                A[i,j] -= s
                A[i,j] /= A[j,j]
            for j in range(i, N):
                s = 0.0
                parir.label('k')
                for k in range(i):
                    s += A[i,k] * A[k,j]
                A[i,j] -= s

def kernel(A):
    N, N = A.shape
    p = {'k': [parir.threads(128), parir.reduce()]}
    lu_parir(A, N, parallelize=p)
