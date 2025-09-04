import math
import parpy
from parpy.operators import inf, max
import torch

@parpy.jit
def relu_kernel(x):
    parpy.label('N')
    x[:] = parpy.operators.max(x[:], 0.0)

def relu(x):
    N = math.prod(x.shape)
    x_flat = x.reshape(N)
    relu_kernel(x_flat, opts=parpy.par({'N': parpy.threads(N)}))
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
                            for e in range(C_out):
                                output[a,i,j,e] += inputs[a,i+b,j+c,d] * weights[b,c,d,e]

@parpy.jit
def add_elemwise(conv2d_res, bias, C_out):
    parpy.label('i')
    for i in range(C_out):
        parpy.label('j')
        parpy.label('k')
        parpy.label('l')
        conv2d_res[:,:,:,i] += bias[i]

# Deep learning convolutional operator (stride = 1)
def conv2d_bias(input, weights, bias):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_in = input.shape[3]
    C_out = weights.shape[3]
    output = parpy.buffer.empty((N, H_out, W_out, C_out), parpy.types.F32, weights.backend)
    p = {'i': parpy.threads(H_out), 'j': parpy.threads(W_out)}
    conv2d_kernel(input, weights, output, H_out, W_out, N, C_in, C_out, K, opts=parpy.par(p))
    p = {'i': parpy.threads(C_out), 'j': parpy.threads(N), 'k': parpy.threads(H_out), 'l': parpy.threads(W_out)}
    add_elemwise(output, bias, C_out, opts=parpy.par(p))
    return output


@parpy.jit
def maxpool2d_kernel(x, output, N_0, N_1, N_2, N_3):
    parpy.label('i')
    for i in range(N_1):
        parpy.label('j')
        for j in range(N_2):
            output[:,i,j,:] = -inf
            for a in range(N_0):
                for b in range(N_3):
                    for ii in range(2):
                        for jj in range(2):
                            output[a,i,j,b] = max(output[a,i,j,b], x[a,2*i+ii,2*j+jj,b])
            

# 2x2 maxpool operator, as used in LeNet-5
def maxpool2d(x):
    output = parpy.buffer.empty(
        (x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]),
        x.dtype, x.backend
    )
    N_0, N_1, N_2, N_3 = output.shape
    p = {'i': parpy.threads(N_1), 'j': parpy.threads(N_2)}
    maxpool2d_kernel(x, output, N_0, N_1, N_2, N_3, opts=parpy.par(p))
    return output


# LeNet-5 Convolutional Neural Network (inference mode)
def lenet5(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b,
           fc3w, fc3b, N, C_before_fc1):
    x = relu(conv2d_bias(input, conv1, conv1bias))
    x = maxpool2d(x)
    x = relu(conv2d_bias(x, conv2, conv2bias))
    x = maxpool2d(x)
    # Past this point, we use intermediate Torch data to use their efficient
    # matrix multiplications. However, the majority of time is spent on the
    # above operations.
    x = x.reshape(N, C_before_fc1).torch()
    x = relu(x @ fc1w.torch() + fc1b.torch())
    x = relu(x @ fc2w.torch() + fc2b.torch())
    res = x @ fc3w.torch() + fc3b.torch()
    return res
