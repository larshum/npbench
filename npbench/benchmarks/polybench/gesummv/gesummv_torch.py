import torch


def kernel(alpha, beta, A, B, x):

    return alpha * A @ x + beta * B @ x
