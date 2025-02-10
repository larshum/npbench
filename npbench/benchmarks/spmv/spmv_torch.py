# Sparse Matrix-Vector Multiplication (SpMV)
import torch
import warnings # ignore warning about CSR tensors being in beta state


# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
def spmv(A_row, A_col, A_val, x):
    nrows, = A_row.shape
    A_row = A_row.to(dtype=torch.int32)
    A_col = A_col.to(dtype=torch.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = torch.sparse_csr_tensor(A_row, A_col, A_val, dtype=A_val.dtype, device=A_val.device)
    return A.matmul(x)
