import parpy
from parpy.operators import sqrt, sum

@parpy.jit
def parpy_kernel(A, R, Q, M, N):
    for k in range(N):
        with parpy.gpu:
            parpy.label('i_reduce')
            nrm = sum(A[:,k] * A[:,k])
            R[k,k] = sqrt(nrm)
        parpy.label('i')
        Q[:,k] = A[:,k] / R[k,k]
        parpy.label('j')
        for j in range(k+1, N):
            parpy.label('i_reduce')
            R[k,j] = sum(Q[:,k] * A[:,j])

        parpy.label('j')
        for j in range(k+1, N):
            parpy.label('i')
            A[:,j] -= Q[:,k] * R[k,j]

def kernel(A):
    Q = parpy.buffer.zeros_like(A)
    R = parpy.buffer.zeros((A.shape[1], A.shape[1]), A.dtype, A.backend)

    M, N = A.shape
    p = {
        'i': parpy.threads(M),
        'i_reduce': parpy.threads(128).reduce(),
        'j': parpy.threads(N)
    }
    parpy_kernel(A, R, Q, M, N, opts=parpy.par(p))
    return Q, R
