import numpy as np
import parpy
from parpy.builtin import convert, minimum, maximum
from parpy.types import I64

@parpy.jit
def compute_helper(array_1, array_2, N, M, a, b, c, out):
    parpy.label('i')
    for i in range(N):
        parpy.label('j')
        for j in range(M):
            clamped = minimum(maximum(array_1[i,j], convert(2, I64)), convert(10, I64))
            out[i,j] = clamped * a + array_2[i,j] * b + c

def compute(array_1, array_2, a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    out = parpy.buffer.empty_like(array_1)
    N, M = array_1.shape
    p = {
        'i': parpy.threads(N),
        'j': parpy.threads(1024)
    }
    compute_helper(array_1, array_2, N, M, a, b, c, out, opts=parpy.par(p))
    return out
