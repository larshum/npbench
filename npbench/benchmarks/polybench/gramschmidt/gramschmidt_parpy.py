import parpy

@parpy.jit
def parpy_kernel(A, R, Q, M, N):
    for k in range(N):
        with parpy.gpu:
            parpy.label('i_reduce')
            nrm = parpy.reduce.sum(A[:,k] * A[:,k])
            R[k,k] = parpy.math.sqrt(nrm)
        parpy.label('i')
        Q[:,k] = A[:,k] / R[k,k]
        parpy.label('j')
        for j in range(k+1, N):
            parpy.label('i_reduce')
            R[k,j] = parpy.reduce.sum(Q[:,k] * A[:,j])

        parpy.label('j')
        for j in range(k+1, N):
            parpy.label('i_inner')
            A[:,j] -= Q[:,k] * R[k,j]

def kernel(A):
    Q = parpy.buffer.zeros_like(A)
    R = parpy.buffer.zeros((A.shape[1], A.shape[1]), A.dtype, A.backend())

    M, N = A.shape
    p = {
        'i': parpy.threads(M),
        'i_inner': parpy.threads(128),
        'i_reduce': parpy.threads(128).par_reduction(),
        'j': parpy.threads(N)
    }
    parpy_kernel(A, R, Q, M, N, opts=parpy.par(p))
    return Q, R
