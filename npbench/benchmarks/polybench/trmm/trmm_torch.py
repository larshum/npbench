import torch


def kernel(alpha, A, B):

    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] += torch.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha
