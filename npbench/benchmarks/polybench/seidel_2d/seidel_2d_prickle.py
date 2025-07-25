import prickle
import torch


@prickle.jit
def prickle_kernel(TSTEPS, N, A):
    with prickle.gpu:
        for t in range(0, TSTEPS - 1):
            for i in range(1, N - 1):
                A[i, 1:-1] += (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] +
                               A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] +
                               A[i + 1, 2:])
                for j in range(1, N - 1):
                    A[i, j] += A[i, j - 1]
                    A[i, j] /= 9.0

def kernel(TSTEPS, N, A):
    prickle_kernel(TSTEPS, N, A, opts=prickle.par({}))
