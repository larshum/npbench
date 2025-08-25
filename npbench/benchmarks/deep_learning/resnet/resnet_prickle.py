import numpy as np
import prickle

@prickle.jit
def relu_kernel(x):
    prickle.label('i')
    x[:] = prickle.max(x[:], 0.0)

def relu(x):
    import math
    # Operate on a flattened version of the buffer
    y = x.reshape((math.prod(x.shape),))
    p = {'i': prickle.threads(y.shape[0])}
    relu_kernel(y, opts=prickle.par(p))
    return x

@prickle.jit
def conv2d_kernel(inputs, weights, output, H_out, W_out, N, C_in, C_out, K):
    prickle.label('i')
    for i in range(H_out):
        prickle.label('j')
        for j in range(W_out):
            output[:,i,j,:] = 0.0
            for a in range(N):
                for b in range(K):
                    for c in range(K):
                        for d in range(C_in):
                            for e in range(C_out):
                                output[a,i,j,e] += inputs[a,i+b,j+c,d] * weights[b,c,d,e]

# Deep learning convolutional operator (stride = 1)
def conv2d(input, weights, ty, backend):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_in = input.shape[3]
    C_out = weights.shape[3]
    output = prickle.buffer.empty((N, H_out, W_out, C_out), ty, backend)
    p = {'i': prickle.threads(H_out), 'j': prickle.threads(W_out)}
    conv2d_kernel(
        input, weights, output, H_out, W_out, N, C_in, C_out, K,
        opts=prickle.par(p)
    )
    return output

@prickle.jit
def batchnorm2d_kernel(x, out, eps, N, M, K, L):
    prickle.label('j')
    for j in range(M):
        prickle.label('k')
        for k in range(K):
            prickle.label('l')
            for l in range(L):
                # Compute the mean
                mean = 0.0
                for i in range(N):
                    mean += x[i,j,k,l]
                mean /= prickle.float32(N)

                # Compute the standard deviation based on mean as in NumPy
                std = 0.0
                for i in range(N):
                    std += (x[i,j,k,l]-mean)**2.0
                std = prickle.sqrt(std / prickle.float32(N))

                # Apply the batch normalization operator
                for i in range(N):
                    out[i,j,k,l] = (x[i,j,k,l] - mean) / prickle.sqrt(std + eps)

# Batch normalization operator, as used in ResNet
def batchnorm2d(x, ty, backend):
    eps = 1e-5
    N, M, K, L = x.shape
    out = prickle.buffer.empty((N, M, K, L), ty, backend)
    p = {'j': prickle.threads(M), 'k': prickle.threads(K), 'l': prickle.threads(L)}
    batchnorm2d_kernel(x, out, eps, N, M, K, L, opts=prickle.par(p))
    return out

# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
def resnet_basicblock(input, conv1, conv2, conv3):
    dtype = input.dtype
    backend = input.backend
    # Pad output of first convolution for second convolution
    padded = np.zeros(
        (input.shape[0], input.shape[1] + 2, input.shape[2] + 2, conv1.shape[3]),
        input.dtype.to_numpy()
    )

    padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1, dtype, backend).numpy()
    x = batchnorm2d(padded, dtype, backend)
    x = relu(x)
    x = conv2d(x, conv2, dtype, backend)
    x = batchnorm2d(x, dtype, backend)
    x = relu(x)
    x = conv2d(x, conv3, dtype, backend)
    x = batchnorm2d(x, dtype, backend)
    return relu(x.numpy() + input.numpy())
