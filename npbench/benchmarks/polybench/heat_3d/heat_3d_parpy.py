import parpy
import torch

@parpy.jit
def parpy_kernel(A, B, TSTEPS):
    for t in range(1, TSTEPS):
        parpy.label('i')
        parpy.label('j')
        parpy.label('k')
        B[1:-1, 1:-1,
          1:-1] = (0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] +
                            A[:-2, 1:-1, 1:-1]) + 0.125 *
                   (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] +
                    A[1:-1, :-2, 1:-1]) + 0.125 *
                   (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] +
                    A[1:-1, 1:-1, 0:-2]) + A[1:-1, 1:-1, 1:-1])

        parpy.label('i')
        parpy.label('j')
        parpy.label('k')
        A[1:-1, 1:-1,
          1:-1] = (0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] +
                            B[:-2, 1:-1, 1:-1]) + 0.125 *
                   (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] +
                    B[1:-1, :-2, 1:-1]) + 0.125 *
                   (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] +
                    B[1:-1, 1:-1, 0:-2]) + B[1:-1, 1:-1, 1:-1])

def kernel(TSTEPS, A, B):
    p = {
        'i': parpy.threads(64),
        'j': parpy.threads(64),
        'k': parpy.threads(64)
    }
    parpy_kernel(A, B, TSTEPS, opts=parpy.par(p))
