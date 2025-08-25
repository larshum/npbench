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
    data_t = data.torch_ref()
    float_n = torch.tensor(float(float_n), dtype=data_t.dtype, device=data_t.device)
    mean = torch.mean(data_t, axis=0)
    data_t -= mean
    cov = prickle.buffer.zeros((M, M), data.dtype, data.backend)
    p = {
        'i': prickle.threads(M),
        'j': prickle.threads(256),
    }
    covariance_prickle(cov, data, float_n, M, opts=prickle.par(p))
    return cov
