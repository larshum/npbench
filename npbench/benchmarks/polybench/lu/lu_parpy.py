import parpy
import torch


@parpy.jit
def lu_parpy(A, N):
    with parpy.gpu:
        for i in range(N):
            for j in range(i):
                s = 0.0
                parpy.label('k')
                for k in range(j):
                    s += A[i,k] * A[k,j]
                A[i,j] -= s
                A[i,j] /= A[j,j]
            for j in range(i, N):
                s = 0.0
                parpy.label('k')
                for k in range(i):
                    s += A[i,k] * A[k,j]
                A[i,j] -= s

def kernel(A):
    N, N = A.shape
    p = {'k': parpy.threads(128).reduce()}
    lu_parpy(A, N, opts=parpy.par(p))
