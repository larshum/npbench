import parpy

@parpy.jit
def parpy_kernel(alpha, beta, C, A, B, M, N):
    parpy.label('M')
    parpy.label('N')
    C[:,:] *= beta

    for i in range(M):
        parpy.label('N')
        for j in range(N):
            parpy.label('M')
            for k in range(i):
                C[k,j] += alpha * B[i,j] * A[i,k]

        parpy.label('N')
        for j in range(N):
            dot_sum = 0.0
            parpy.label('i_red')
            for k in range(i):
                dot_sum += B[k,j] * A[i,k]
            C[i,j] += alpha * B[i,j] * A[i,i] + alpha * dot_sum

def kernel(alpha, beta, C, A, B):
    M, N = C.shape
    p = {
        'M': parpy.threads(M),
        'N': parpy.threads(N),
        'i_red': parpy.threads(32).par_reduction()
    }
    opts = parpy.par(p)
    opts.max_unroll_count = 0
    parpy_kernel(alpha, beta, C, A, B, M, N, opts=opts)
