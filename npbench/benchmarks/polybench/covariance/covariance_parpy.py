import parpy
from parpy.operators import sum
import torch


@parpy.jit
def covariance_parpy(cov, data, float_n, M):
    parpy.label('i')
    for i in range(M):
        parpy.label('j')
        for j in range(i, M):
            s = sum(data[:, i] * data[:, j])
            cov[i, j] = s / (float_n - 1.0)
            cov[j, i] = cov[i, j]

def kernel(M, float_n, data):
    float_n = torch.tensor(float(float_n), dtype=data.dtype, device=data.device)
    mean = torch.mean(data, axis=0)
    data -= mean
    cov = torch.zeros((M, M), dtype=data.dtype, device=data.device)
    p = {
        'i': parpy.threads(M),
        'j': parpy.threads(256),
    }
    covariance_parpy(cov, data, float_n, M, opts=parpy.par(p))
    return cov
