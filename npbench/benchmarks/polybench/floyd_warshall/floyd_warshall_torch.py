import torch


def kernel(path):

    for k in range(path.shape[0]):
        path[:] = torch.minimum(path[:], path[:, k].reshape(-1, 1) + path[k, :])

kernel_jit = torch.compile(kernel)
