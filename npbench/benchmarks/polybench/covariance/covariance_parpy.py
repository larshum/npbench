import parpy
import torch

@parpy.jit
def covariance_parpy(cov, data, float_n, M):
    parpy.label('i')
    for i in range(M):
        parpy.label('j')
        for j in range(i, M):
            s = parpy.reduce.sum(data[:, i] * data[:, j])
            cov[i, j] = s / (float_n - 1.0)
            cov[j, i] = cov[i, j]

def kernel(M, float_n, data):
    t_data = data.torch()
    float_n = torch.tensor(float(float_n), dtype=t_data.dtype, device=t_data.device)
    mean = torch.mean(t_data, axis=0)
    t_data -= mean
    cov = parpy.buffer.zeros((M, M), data.dtype, data.backend())
    p = {
        'i': parpy.threads(M),
        'j': parpy.threads(256),
    }
    opts = parpy.par(p)
    opts.max_unroll_count = 0
    covariance_parpy(cov, t_data, float_n, M, opts=opts)
    return cov
