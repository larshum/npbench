import prickle

@prickle.jit
def kernel_wrap(A, B, TSTEPS):
    for t in range(1, TSTEPS):
        prickle.label('i')
        prickle.label('j')
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                               A[2:, 1:-1] + A[:-2, 1:-1])

        prickle.label('i')
        prickle.label('j')
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                               B[2:, 1:-1] + B[:-2, 1:-1])


def kernel(TSTEPS, A, B):
    N, N = A.shape
    p = {
        'i': prickle.threads(N-1),
        'j': prickle.threads(N-1),
    }
    kernel_wrap(A, B, TSTEPS, opts=prickle.par(p))
