import torch


def kernel(A, x):

    return (A @ x) @ A

kernel_jit = torch.compile(kernel)
