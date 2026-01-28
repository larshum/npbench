import math
import parpy
import torch

@parpy.jit
def relu_kernel(x):
    parpy.label('N')
    x[:] = parpy.builtin.maximum(x[:], 0.0)

def relu(x):
    N = math.prod(x.shape)
    x_flat = x.reshape(N)
    opts = parpy.par({'N': parpy.threads(N)})
    opts.max_unroll_count = 0
    relu_kernel(x_flat, opts=opts)
    return x

@parpy.jit
def conv2d_kernel(inputs, weights, output, H_out, W_out, N, C_in, C_out, K):
    parpy.label('i')
    for i in range(H_out):
        parpy.label('j')
        for j in range(W_out):
            output[:,i,j,:] = 0.0
            for a in range(N):
                for b in range(K):
                    for c in range(K):
                        for d in range(C_in):
                            parpy.label('k')
                            for e in range(C_out):
                                output[a,i,j,e] += inputs[a,i+b,j+c,d] * weights[b,c,d,e]

# Deep learning convolutional operator (stride = 1)
def conv2d(input, weights):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_in = input.shape[3]
    C_out = weights.shape[3]
    output = parpy.buffer.empty((N, H_out, W_out, C_out), parpy.types.F32, weights.backend())
    p = {'i': parpy.threads(H_out), 'j': parpy.threads(W_out), 'k': parpy.threads(C_out)}
    opts = parpy.par(p)
    opts.max_unroll_count = 0
    conv2d_kernel(input, weights, output, H_out, W_out, N, C_in, C_out, K, opts=opts)
    return output

@parpy.jit
def batchnorm2d_kernel(x, mean, std, eps, out, N):
    for i in range(N):
        parpy.label('j')
        parpy.label('k')
        parpy.label('l')
        mean[0,:,:,:] += x[i,:,:,:]

    parpy.label('j')
    parpy.label('k')
    parpy.label('l')
    mean[0,:,:,:] /= parpy.builtin.convert(N, parpy.types.F32)

    for i in range(N):
        parpy.label('j')
        parpy.label('k')
        parpy.label('l')
        std[0,:,:,:] += (x[i,:,:,:] - mean[0,:,:,:]) ** 2.0

    parpy.label('j')
    parpy.label('k')
    parpy.label('l')
    std[0,:,:,:] = parpy.math.sqrt(std[0,:,:,:] / parpy.builtin.convert(N, parpy.types.F32))

    parpy.label('i')
    for i in range(N):
        parpy.label('j')
        parpy.label('k')
        parpy.label('l')
        out[i,:,:,:] = (x[i,:,:,:] - mean[0,:,:,:]) / parpy.math.sqrt(std[0,:,:,:] - eps)

# Batch normalization operator, as used in ResNet
def batchnorm2d(x, eps=1e-5):
    reduced_shape = list(x.shape)
    N = reduced_shape[0]
    reduced_shape[0] = 1
    mean = parpy.buffer.zeros(reduced_shape, x.dtype, x.backend())
    std = parpy.buffer.zeros(reduced_shape, x.dtype, x.backend())
    out = parpy.buffer.empty_like(x)
    p = {
        'i': parpy.threads(N),
        'j': parpy.threads(reduced_shape[1]),
        'k': parpy.threads(reduced_shape[2]),
        'l': parpy.threads(reduced_shape[3]),
    }
    opts = parpy.par(p)
    opts.max_unroll_count = 0
    batchnorm2d_kernel(x, mean, std, eps, out, N, opts=opts)
    return out

@parpy.jit
def flip_kernel(tmp, padded):
    parpy.label('i')
    parpy.label('j')
    parpy.label('k')
    parpy.label('l')
    padded[:, 1:-1, 1:-1, :] = tmp[:, :, :, :]

def flip(tmp, padded):
    I, J, K, L = tmp.shape
    p = {
        'i': parpy.threads(I),
        'j': parpy.threads(J),
        'k': parpy.threads(K),
        'l': parpy.threads(L),
    }
    opts = parpy.par(p)
    opts.max_unroll_count = 0
    flip_kernel(tmp, padded, opts=opts)

@parpy.jit
def add_elemwise_kernel(x, y, out):
    parpy.label('N')
    out[:] = x[:] + y[:]

def add_elemwise(x, input):
    N = math.prod(x.shape)
    x_flat = x.reshape(N)
    input_flat = input.reshape(N)
    out = parpy.buffer.empty_like(input)
    out_flat = out.reshape(N)
    opts = parpy.par({'N': parpy.threads(N)})
    opts.max_unroll_count = 0
    add_elemwise_kernel(x_flat, input_flat, out_flat, opts=opts)
    return out

# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
def resnet_basicblock(input, conv1, conv2, conv3):
    # Pad output of first convolution for second convolution
    padded = parpy.buffer.zeros(
        (input.shape[0], input.shape[1] + 2, input.shape[2] + 2, conv1.shape[3]),
        input.dtype, input.backend()
    )

    # padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1)
    tmp = conv2d(input, conv1)
    flip(tmp, padded)

    x = batchnorm2d(padded)
    x = relu(x)

    x = conv2d(x, conv2)
    x = batchnorm2d(x)
    x = relu(x)
    x = conv2d(x, conv3)
    x = batchnorm2d(x)
    return relu(add_elemwise(x, input))
