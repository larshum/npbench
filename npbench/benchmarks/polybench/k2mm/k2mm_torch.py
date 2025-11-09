import torch


def kernel(alpha, beta, A, B, C, D):

    D[:] = alpha * A @ B @ C + beta * D

kernel_jit = torch.compile(kernel)
