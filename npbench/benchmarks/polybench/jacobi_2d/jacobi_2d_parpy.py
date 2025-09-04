import parpy

@parpy.jit
def kernel_wrap(A, B, TSTEPS):
    for t in range(1, TSTEPS):
        parpy.label('i')
        parpy.label('j')
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                               A[2:, 1:-1] + A[:-2, 1:-1])

        parpy.label('i')
        parpy.label('j')
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                               B[2:, 1:-1] + B[:-2, 1:-1])


def kernel(TSTEPS, A, B):
    N, N = A.shape
    p = {
        'i': parpy.threads(N-1),
        'j': parpy.threads(N-1),
    }
    kernel_wrap(A, B, TSTEPS, opts=parpy.par(p))
