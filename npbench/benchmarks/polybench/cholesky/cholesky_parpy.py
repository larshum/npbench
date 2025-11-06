import parpy

@parpy.jit
def parpy_kernel(A, N):
    with parpy.gpu:
        A[0,0] = parpy.math.sqrt(A[0,0])
        for i in range(1, N):
            for j in range(i):
                s = 0.0
                parpy.label('k')
                for k in range(j):
                    s += A[i,k] * A[j,k]
                A[i,j] -= s
                A[i,j] = A[i,j] / A[j,j]
            s = 0.0
            for k in range(i):
                s += A[i,k] * A[i,k]
            A[i,i] -= s
            A[i,i] = parpy.math.sqrt(A[i,i])

def kernel(A):
    N, _ = A.shape
    p = { 'k': parpy.threads(256).par_reduction() }
    parpy_kernel(A, N, opts=parpy.par(p))
