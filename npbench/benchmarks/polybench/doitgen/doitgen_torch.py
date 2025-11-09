import torch


def kernel(NR, NQ, NP, A, C4):
    A[:] = torch.reshape(torch.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))

kernel_jit = torch.compile(kernel)
