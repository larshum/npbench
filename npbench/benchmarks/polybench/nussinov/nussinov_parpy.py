import parpy
from parpy.builtin import maximum

@parpy.jit
def parpy_kernel(table, seq, N):
    with parpy.gpu:
        for i in range(N-1, -1, -1):
            for j in range(i+1, N):
                if j-1 >= 0:
                    table[i,j] = maximum(table[i,j], table[i,j-1])
                if i+1 < N:
                    table[i,j] = maximum(table[i,j], table[i+1,j])
                if j-1 >= 0 and i+1 < N:
                    if i < j-1:
                        m = 1 if seq[i] + seq[j] == 3 else 0
                        table[i,j] = maximum(table[i,j], table[i+1,j-1] + m)
                    else:
                        table[i,j] = maximum(table[i,j], table[i+1,j-1])
                for k in range(i+1, j):
                    table[i,j] = maximum(table[i,j], table[i,k] + table[k+1,j])

def kernel(N, seq):
    table = parpy.buffer.zeros((N, N), parpy.types.I32, seq.backend())
    parpy_kernel(table, seq, N, opts=parpy.par({}))
    return table
