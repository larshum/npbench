import numpy as np
import torch


def kernel(M, float_n, data):
    float_n = torch.tensor(float_n, dtype=torch.float64, device=data.device)
    mean = torch.mean(data, axis=0)
    # NOTE: Translating to NumPy and back for precision as torch.std fails validation
    stddev = torch.tensor(np.std(data.cpu().numpy(), axis=0), dtype=torch.float64, device=data.device)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= torch.sqrt(float_n) * stddev
    corr = torch.eye(M, dtype=data.dtype, device=data.device)
    for i in range(M - 1):
        corr[i + 1:M, i] = corr[i, i + 1:M] = data[:, i] @ data[:, i + 1:M]

    return corr
