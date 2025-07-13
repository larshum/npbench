import prickle
import torch


@prickle.jit
def prickle_kernel(corr, data, M):
    prickle.label('i')
    for i in range(M-1):
        prickle.label('j')
        for j in range(i+1, M):
            corr[i, j] = prickle.sum(data[:, i] * data[:, j])
            corr[j, i] = corr[i, j]

def kernel(M, float_n, data):
    float_n = torch.tensor(float(float_n), dtype=data.dtype, device=data.device)
    mean = torch.mean(data, axis=0)
    stddev = torch.std(data, unbiased=False, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= torch.sqrt(float_n) * stddev
    corr = torch.eye(M, dtype=data.dtype, device=data.device)
    p = {
        'i': prickle.threads(M-1),
        'j': prickle.threads(256),
    }
    prickle_kernel(corr, data, M, opts=prickle.par(p))
    return corr
