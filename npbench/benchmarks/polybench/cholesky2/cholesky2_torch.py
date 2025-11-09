import torch


def kernel(A):
    A[:] = torch.linalg.cholesky(A) + torch.triu(A, 1)

kernel_jit = torch.compile(kernel)
