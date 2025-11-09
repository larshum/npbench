import torch


def relu(x):
    return torch.maximum(x, torch.zeros_like(x))


# Numerically-stable version of softmax
def softmax(x):
    tmp_max = torch.max(x, axis=-1, keepdims=True)
    tmp_out = torch.exp(x - tmp_max.values)
    tmp_sum = torch.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


# 3-layer MLP
def mlp(input, w1, b1, w2, b2, w3, b3):
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    x = softmax(x @ w3 + b3)  # Softmax call can be omitted if necessary
    return x

mlp_jit = torch.compile(mlp)
