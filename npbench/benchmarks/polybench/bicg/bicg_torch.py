import torch


def kernel(A, p, r):

    return r @ A, A @ p

kernel_jit = torch.compile(kernel)
