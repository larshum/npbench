import prickle
import torch


@prickle.jit
def prickle_kernel(r, y, temp, N):
    with prickle.gpu:
        alpha = -r[0]
        beta = 1.0
        y[0] = -r[0]
        for k in range(1, N):
            beta *= 1.0 - alpha * alpha
            t = 0.0
            prickle.label('k_red')
            for i in range(k):
                t += r[k-i-1] * y[i]
            alpha = -(r[k] + t) / beta
            prickle.label('k')
            for i in range(k):
                temp[i] = alpha * y[k-i-1]
            prickle.label('k')
            for i in range(k):
                y[i] += temp[i]
            y[k] = alpha


def kernel(r):
    y = torch.empty_like(r)
    temp = torch.empty_like(y)
    N, = r.shape
    p = {
        'k_red': prickle.threads(512).reduce(),
        'k': prickle.threads(512)
    }
    prickle_kernel(r, y, temp, N, opts=prickle.par(p))
    return y

    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, r.shape[0]):
        beta *= 1.0 - alpha * alpha
        alpha = -(r[k] + torch.dot(torch.flip(r[:k], dims=[0]), y[:k])) / beta
        y[:k] += alpha * torch.flip(y[:k], dims=[0])
        y[k] = alpha

    return y
