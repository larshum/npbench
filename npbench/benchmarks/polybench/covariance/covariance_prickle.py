import prickle
import torch


@prickle.jit
def covariance_prickle(cov, data, float_n, M):
    prickle.label('i')
    for i in range(M):
        prickle.label('j')
        for j in range(i, M):
            s = prickle.sum(data[:, i] * data[:, j])
            cov[i, j] = s / (float_n - 1.0)
            cov[j, i] = cov[i, j]

def kernel(M, float_n, data):
    float_n = torch.tensor(float(float_n), dtype=data.dtype, device=data.device)
    mean = torch.mean(data, axis=0)
    data -= mean
    cov = torch.zeros((M, M), dtype=data.dtype, device=data.device)
    p = {
        'i': prickle.threads(M),
        'j': prickle.threads(256),
    }
    covariance_prickle(cov, data, float_n, M, opts=prickle.par(p))
    return cov
