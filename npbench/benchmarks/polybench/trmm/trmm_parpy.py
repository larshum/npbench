import parpy

@parpy.jit
def trmm(alpha, A, B, M, N):
    for i in range(M):
        parpy.label('j')
        for j in range(N):
            parpy.label('k')
            for k in range(i+1, M):
                B[i,j] += A[k,i] * B[k,j]
            B[i,j] = B[i,j] * alpha

def kernel(alpha, A, B):
    M, N = B.shape
    p = {
        'j': parpy.threads(N),
        'k': parpy.threads(256).par_reduction()
    }
    trmm(alpha, A, B, M, N, opts=parpy.par(p))
