# Sparse Matrix-Vector Multiplication (SpMV)
import prickle

@prickle.jit
def spmv_helper(A_row, A_col, A_val, N, x, y):
    prickle.label('i')
    for i in range(N - 1):
        prickle.label('j')
        for j in range(A_row[i], A_row[i+1]):
            y[i] += A_val[j] * x[A_col[j]]

# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
def spmv(A_row, A_col, A_val, x):
    N, = A_row.shape
    i32 = prickle.buffer.DataType("<i4")
    A_row = A_row.with_type(i32)
    A_col = A_col.with_type(i32)
    y = prickle.buffer.zeros((N - 1,), A_val.dtype, A_val.backend)
    p = {
        'i': prickle.threads(N-1),
        'j': prickle.threads(64).reduce(),
    }
    spmv_helper(A_row, A_col, A_val, N, x, y, opts=prickle.par(p))
    return y
