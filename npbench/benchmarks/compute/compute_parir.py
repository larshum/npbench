import parir
import torch

@parir.jit
def compute_helper(array_1, array_2, N, M, a, b, c, out):
    parir.label('i')
    for i in range(N):
        parir.label('j')
        for j in range(M):
            clamped = parir.min(parir.max(array_1[i,j], parir.int64(2)), parir.int64(10))
            out[i,j] = clamped * a + array_2[i,j] * b + c

def compute(array_1, array_2, a, b, c):
    a = torch.tensor(a)
    b = torch.tensor(b)
    c = torch.tensor(c)
    out = torch.zeros_like(array_1)
    N, M = array_1.shape
    p = {
        'i': parir.threads(N),
        'j': parir.threads(1024)
    }
    compute_helper(array_1, array_2, N, M, a, b, c, out, opts=parir.par(p))
    return out
