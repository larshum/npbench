import parir
from parir import ParKind
import torch

@parir.jit
def syrk(alpha, beta, C, A, N, M):
    for i in range(N):
        for j in range(i+1):
            C[i,j] = C[i,j] * beta[0]
            for k in range(M):
                C[i,j] = C[i,j] + alpha[0] * A[i,k] * A[j,k]

fn = None

def kernel(alpha, beta, C, A):
    global fn
    alpha = torch.tensor([alpha], dtype=torch.float64, device='cuda')
    beta = torch.tensor([beta], dtype=torch.float64, device='cuda')
    N, M = A.shape
    p = {
        'i': [ParKind.GpuThreads(N)],
        'k': [ParKind.GpuThreads(256), ParKind.GpuReduction()]
    }
    syrk(alpha, beta, C, A, N, M, parallelize=p)

