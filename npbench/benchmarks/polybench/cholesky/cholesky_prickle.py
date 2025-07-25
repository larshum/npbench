import prickle
import torch


@prickle.jit
def prickle_kernel(A, N):
    with prickle.gpu:
        A[0,0] = prickle.sqrt(A[0,0])
        for i in range(1, N):
            for j in range(i):
                s = 0.0
                prickle.label('k')
                for k in range(j):
                    s += A[i,k] * A[j,k]
                A[i,j] -= s
                A[i,j] = A[i,j] / A[j,j]
            s = 0.0
            for k in range(i):
                s += A[i,k] * A[i,k]
            A[i,i] -= s
            A[i,i] = prickle.sqrt(A[i,i])

def kernel(A):
    N, _ = A.shape
    p = { 'k': prickle.threads(256).reduce() }
    prickle_kernel(A, N, opts=prickle.par(p))
