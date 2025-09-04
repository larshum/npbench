# Sparse Matrix-Vector Multiplication (SpMV)
import parpy

@parpy.jit
def spmv_helper(A_row, A_col, A_val, N, x, y):
    parpy.label('i')
    for i in range(N - 1):
        parpy.label('j')
        for j in range(A_row[i], A_row[i+1]):
            y[i] += A_val[j] * x[A_col[j]]

# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
def spmv(A_row, A_col, A_val, x):
    N, = A_row.shape
    A_row = A_row.with_type(parpy.types.I32)
    A_col = A_col.with_type(parpy.types.I32)
    y = parpy.buffer.zeros((N-1,), A_val.dtype, A_val.backend)
    p = {
        'i': parpy.threads(N-1),
        'j': parpy.threads(64).reduce(),
    }
    spmv_helper(A_row, A_col, A_val, N, x, y, opts=parpy.par(p))
    return y
