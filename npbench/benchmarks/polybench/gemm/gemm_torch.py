import torch


def kernel(alpha, beta, C, A, B):

    C[:] = alpha * A @ B + beta * C

kernel_jit = torch.compile(kernel)
