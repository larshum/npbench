import torch


def kernel(A):

    A[0, 0] = torch.sqrt(A[0, 0])
    for i in range(1, A.shape[0]):
        for j in range(i):
            A[i, j] -= torch.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= torch.dot(A[i, :i], A[i, :i])
        A[i, i] = torch.sqrt(A[i, i])
