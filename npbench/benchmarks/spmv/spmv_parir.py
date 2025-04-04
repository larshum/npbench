# Sparse Matrix-Vector Multiplication (SpMV)
import parir
import torch

@parir.jit
def spmv_helper(A_row, A_col, A_val, N, x, y):
    parir.label('i')
    for i in range(N - 1):
        parir.label('j')
        for j in range(A_row[i], A_row[i+1]):
            y[i] += A_val[j] * x[A_col[j]]

# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
def spmv(A_row, A_col, A_val, x):
    N, = A_row.shape
    A_row = A_row.to(dtype=torch.int64)
    A_col = A_col.to(dtype=torch.int64)
    y = torch.zeros(N - 1, dtype=A_val.dtype, device='cuda')
    p = {
        'i': parir.threads(N-1),
        'j': parir.threads(64).reduce()
    }
    spmv_helper(A_row, A_col, A_val, N, x, y, parallelize=p)
    return y
