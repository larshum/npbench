import numpy as np
import parpy
from parpy.operators import int64, min, max

@parpy.jit
def compute_helper(array_1, array_2, N, M, a, b, c, out):
    parpy.label('i')
    for i in range(N):
        parpy.label('j')
        for j in range(M):
            clamped = min(max(array_1[i,j], int64(2)), int64(10))
            out[i,j] = clamped * a + array_2[i,j] * b + c

def compute(array_1, array_2, a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    out = parpy.buffer.zeros_like(array_1)
    N, M = array_1.shape
    p = {
        'i': parpy.threads(N),
        'j': parpy.threads(1024)
    }
    compute_helper(array_1, array_2, N, M, a, b, c, out, opts=parpy.par(p))
    return out
