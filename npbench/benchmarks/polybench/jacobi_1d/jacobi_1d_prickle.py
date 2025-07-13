import prickle
import torch

@prickle.jit
def kernel_wrap(A, B, TSTEPS):
    for t in range(1, TSTEPS):
        prickle.label('i')
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])

        prickle.label('i')
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])

def kernel(TSTEPS, A, B):
    N, = A.shape
    p = {'i': prickle.threads(N-2)}
    kernel_wrap(A, B, TSTEPS, opts=prickle.par(p))
