import parpy

@parpy.jit
def kernel_wrap(A, B, TSTEPS):
    for t in range(1, TSTEPS):
        parpy.label('i')
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])

        parpy.label('i')
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])

def kernel(TSTEPS, A, B):
    N, = A.shape
    p = {'i': parpy.threads(N-2)}
    opts = parpy.par(p)
    opts.max_unroll_count = 0
    kernel_wrap(A, B, TSTEPS, opts=opts)
