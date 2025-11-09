import torch


# Numerically-stable version of softmax
def softmax(x):
    tmp_max = torch.max(x, axis=-1, keepdims=True)
    tmp_out = torch.exp(x - tmp_max.values)
    tmp_sum = torch.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum

softmax_jit = torch.compile(softmax)
