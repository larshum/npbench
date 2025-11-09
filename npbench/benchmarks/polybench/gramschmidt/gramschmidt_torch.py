import torch


def kernel(A):

    Q = torch.zeros_like(A)
    R = torch.zeros((A.shape[1], A.shape[1]), dtype=A.dtype, device=A.device)

    for k in range(A.shape[1]):
        nrm = torch.dot(A[:, k], A[:, k])
        R[k, k] = torch.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, A.shape[1]):
            R[k, j] = torch.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]

    return Q, R

kernel_jit = torch.compile(kernel)
