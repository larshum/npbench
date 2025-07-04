import parir
import torch


@parir.jit
def covariance_parir(cov, data, float_n, M):
    parir.label('i')
    for i in range(M):
        parir.label('j')
        for j in range(i, M):
            s = parir.sum(data[:, i] * data[:, j])
            cov[i, j] = s / (float_n - 1.0)
            cov[j, i] = cov[i, j]

def kernel(M, float_n, data):
    float_n = torch.tensor(float(float_n), dtype=data.dtype, device=data.device)
    mean = torch.mean(data, axis=0)
    data -= mean
    cov = torch.zeros((M, M), dtype=data.dtype, device=data.device)
    p = {
        'i': parir.threads(M),
        'j': parir.threads(256),
    }
    covariance_parir(cov, data, float_n, M, opts=parir.par(p))
    return cov
