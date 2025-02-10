import torch


def kernel(A):
    A[:] = torch.linalg.cholesky(A) + torch.triu(A, 1)
