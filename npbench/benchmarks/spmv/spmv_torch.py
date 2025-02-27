# Sparse Matrix-Vector Multiplication (SpMV)
import torch


# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
def spmv(A_row, A_col, A_val, x):
    A_row = A_row.to(dtype=torch.int32)
    A_col = A_col.to(dtype=torch.int32)
    y = torch.empty(A_row.shape[0] - 1, dtype=A_val.dtype, device=A_val.device)

    for i in range(A_row.shape[0] - 1):
        cols = A_col[A_row[i]:A_row[i + 1]]
        vals = A_val[A_row[i]:A_row[i + 1]]
        y[i] = vals @ x[cols]

    return y
