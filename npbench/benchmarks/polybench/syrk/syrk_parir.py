import parir
import torch

@parir.jit
def syrk(alpha, beta, C, A, N, M):
    parir.label('i')
    for i in range(N):
        for j in range(i+1):
            C[i,j] *= beta[0]
            parir.label('k')
            for k in range(M):
                C[i,j] += alpha[0] * A[i,k] * A[j,k]

fn = None

def kernel(alpha, beta, C, A):
    global fn
    alpha = torch.tensor([alpha], dtype=torch.float64, device='cuda')
    beta = torch.tensor([beta], dtype=torch.float64, device='cuda')
    N, M = A.shape
    p = {
        'i': [parir.threads(N)],
        'k': [parir.threads(256), parir.reduce()]
    }
    syrk(alpha, beta, C, A, N, M, parallelize=p)

