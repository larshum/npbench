import torch


def relu(x):
    return torch.maximum(x, torch.zeros_like(x))


# Deep learning convolutional operator (stride = 1)
def conv2d(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_out = weights.shape[3]
    output = torch.empty((N, H_out, W_out, C_out), dtype=torch.float32, device=weights.device)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    for i in range(H_out):
        for j in range(W_out):
            output[:, i, j, :] = torch.sum(
                input[:, i:i + K, j:j + K, :, torch.newaxis] *
                weights[torch.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


# Batch normalization operator, as used in ResNet
def batchnorm2d(x, eps=1e-5):
    mean = torch.mean(x, axis=0, keepdims=True)
    std = torch.std(x, axis=0, unbiased=False, keepdims=True)
    return (x - mean) / torch.sqrt(std + eps)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
def resnet_basicblock(input, conv1, conv2, conv3):
    # Pad output of first convolution for second convolution
    padded = torch.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2,
                       conv1.shape[3]), device=input.device)

    padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1)
    x = batchnorm2d(padded)
    x = relu(x)

    x = conv2d(x, conv2)
    x = batchnorm2d(x)
    x = relu(x)
    x = conv2d(x, conv3)
    x = batchnorm2d(x)
    return relu(x + input)

resnet_basicblock_jit = torch.compile(resnet_basicblock)
