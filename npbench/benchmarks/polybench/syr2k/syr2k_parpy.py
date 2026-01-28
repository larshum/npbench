import parpy

@parpy.jit
def kernel_wrap(alpha, beta, C, A, B, N, M):
    parpy.label('i')
    for i in range(N):
        for j in range(i+1):
            C[i,j] *= beta
            parpy.label('k')
            for k in range(M):
                C[i,j] += A[j,k] * alpha * B[i,k] + B[j,k] * alpha * A[i,k]

def kernel(alpha, beta, C, A, B):
    N, M = A.shape
    p = {
        'i': parpy.threads(N),
        'k': parpy.threads(256).par_reduction()
    }
    opts = parpy.par(p)
    opts.max_unroll_count = 0
    kernel_wrap(alpha, beta, C, A, B, N, M, opts=opts)

