import parir
import torch

@parir.jit
def parir_kernel(A, R, Q, M, N):
    for k in range(N):
        with parir.gpu:
            parir.label('i_reduce')
            nrm = parir.sum(A[:,k] * A[:,k])
            R[k,k] = parir.sqrt(nrm)
        parir.label('i')
        Q[:,k] = A[:,k] / R[k,k]
        parir.label('j')
        for j in range(k+1, N):
            parir.label('i_reduce')
            R[k,j] = parir.sum(Q[:,k] * A[:,j])

        parir.label('j')
        for j in range(k+1, N):
            parir.label('i')
            A[:,j] -= Q[:,k] * R[k,j]

def kernel(A):

    Q = torch.zeros_like(A)
    R = torch.zeros((A.shape[1], A.shape[1]), dtype=A.dtype, device=A.device)

    M, N = A.shape
    p = {
        'i': parir.threads(M),
        'i_reduce': parir.threads(128).reduce(),
        'j': parir.threads(N)
    }
    parir_kernel(A, R, Q, M, N, opts=parir.parallelize(p))
    return Q, R
