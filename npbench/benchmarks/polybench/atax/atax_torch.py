import torch


def kernel(A, x):

    return (A @ x) @ A
