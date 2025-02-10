import torch


def kernel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):

    A += torch.outer(u1, v1) + torch.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x
