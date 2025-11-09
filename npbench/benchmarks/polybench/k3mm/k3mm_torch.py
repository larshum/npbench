import torch


def kernel(A, B, C, D):

    return A @ B @ C @ D

kernel_jit = torch.compile(kernel)
