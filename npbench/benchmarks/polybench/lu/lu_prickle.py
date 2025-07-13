import prickle
import torch


@prickle.jit
def lu_prickle(A, N):
    with prickle.gpu:
        for i in range(N):
            for j in range(i):
                s = 0.0
                prickle.label('k')
                for k in range(j):
                    s += A[i,k] * A[k,j]
                A[i,j] -= s
                A[i,j] /= A[j,j]
            for j in range(i, N):
                s = 0.0
                prickle.label('k')
                for k in range(i):
                    s += A[i,k] * A[k,j]
                A[i,j] -= s

def kernel(A):
    N, N = A.shape
    p = {'k': prickle.threads(128).reduce()}
    lu_prickle(A, N, opts=prickle.par(p))
