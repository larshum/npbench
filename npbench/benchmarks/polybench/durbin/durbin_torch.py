import torch


def kernel(r):

    y = torch.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, r.shape[0]):
        beta *= 1.0 - alpha * alpha
        alpha = -(r[k] + torch.dot(torch.flip(r[:k], dims=[0]), y[:k])) / beta
        y[:k] += alpha * torch.flip(y[:k], dims=[0])
        y[k] = alpha

    return y
