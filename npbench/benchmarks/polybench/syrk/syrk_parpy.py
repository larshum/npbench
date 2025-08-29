import parpy
import torch

@parpy.jit
def syrk(alpha, beta, C, A, N, M):
    parpy.label('i')
    for i in range(N):
        for j in range(i+1):
            C[i,j] *= beta[0]
            parpy.label('k')
            for k in range(M):
                C[i,j] += alpha[0] * A[i,k] * A[j,k]

def kernel(alpha, beta, C, A):
    alpha = torch.tensor([alpha], dtype=A.dtype)
    beta = torch.tensor([beta], dtype=A.dtype)
    N, M = A.shape
    p = {
        'i': parpy.threads(N),
        'k': parpy.threads(256).reduce()
    }
    syrk(alpha, beta, C, A, N, M, opts=parpy.par(p))

