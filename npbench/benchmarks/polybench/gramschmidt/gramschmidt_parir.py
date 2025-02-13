import parir
import torch

@parir.jit
def parir_kernel(A, R, Q, M, N):
    for k in range(N):
        with parir.gpu:
            nrm = 0.0
            parir.label('i_reduce')
            for i in range(M):
                nrm += A[i,k] * A[i,k]
            R[k,k] = parir.sqrt(nrm)
        parir.label('i')
        for i in range(M):
            Q[i,k] = A[i,k] / R[k,k]
        parir.label('j')
        for j in range(k+1, N):
            parir.label('i_reduce')
            for i in range(M):
                R[k,j] += Q[i,k] * A[i,j]

        parir.label('j')
        for j in range(k+1, N):
            parir.label('i')
            for i in range(M):
                A[i,j] -= Q[i,k] * R[k,j]

def kernel(A):

    Q = torch.zeros_like(A)
    R = torch.zeros((A.shape[1], A.shape[1]), dtype=A.dtype, device=A.device)

    M, N = A.shape
    p = {
        'i': [parir.threads(M)],
        'i_reduce': [parir.threads(128), parir.reduce()],
        'j': [parir.threads(N)]
    }
    parir_kernel(A, R, Q, M, N, parallelize=p)
    return Q, R
