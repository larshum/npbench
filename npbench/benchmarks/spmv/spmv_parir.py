# Sparse Matrix-Vector Multiplication (SpMV)
import parir
from parir import ParKind
import torch

@parir.jit
def spmv_helper(A_row, A_col, A_val, N, x, y):
    for i in range(N - 1):
        for j in range(A_row[i], A_row[i+1]):
            y[i] = y[i] + A_val[j] * x[A_col[j]]

# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
def spmv(A_row, A_col, A_val, x):
    N, = A_row.shape
    A_row = A_row.to(dtype=torch.int64)
    A_col = A_col.to(dtype=torch.int64)
    y = torch.zeros(N - 1, dtype=A_val.dtype, device='cuda')
    p = {
        'i': [ParKind.GpuThreads(N-1)],
        'j': [ParKind.GpuThreads(64), ParKind.GpuReduction()],
    }
    spmv_helper(A_row, A_col, A_val, N, x, y, parallelize=p)
    return y.cpu().numpy()
