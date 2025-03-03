import parir
import torch


@parir.jit
def parir_kernel(corr, data, M):
    parir.label('i')
    for i in range(M-1):
        parir.label('j')
        for j in range(i+1, M):
            parir.label('reduce')
            corr[i, j] = parir.sum(data[:, i] * data[:, j])
            corr[j, i] = corr[i, j]

def kernel(M, float_n, data):
    float_n = torch.tensor(float(float_n), dtype=torch.float64, device=data.device)
    mean = torch.mean(data, axis=0)
    stddev = torch.std(data, unbiased=False, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= torch.sqrt(float_n) * stddev
    corr = torch.eye(M, dtype=data.dtype, device=data.device)
    p = {
        'i': [parir.threads(M-1)],
        'j': [parir.threads(M//2)],
        'reduce': [parir.threads(1024)]
    }
    parir_kernel(corr, data, M, parallelize=p)
    return corr
