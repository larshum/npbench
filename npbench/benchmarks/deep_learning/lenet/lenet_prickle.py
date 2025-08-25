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
def conv2d_kernel(inputs, weights, output, bias, H_out, W_out, N, C_in, C_out, K):
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

    prickle.label('i')
    for i in range(H_out):
        prickle.label('j')
        for j in range(W_out):
            for a in range(N):
                output[a,i,j,:] += bias[:]

# Deep learning convolutional operator (stride = 1)
def conv2d(input, weights, bias):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_in = input.shape[3]
    C_out = weights.shape[3]
    output = prickle.buffer.empty((N, H_out, W_out, C_out), input.dtype, weights.backend)
    p = {'i': prickle.threads(H_out), 'j': prickle.threads(W_out)}
    conv2d_kernel(
        input, weights, output, bias, H_out, W_out, N, C_in, C_out, K,
        opts=prickle.par(p)
    )
    return output

@prickle.jit
def maxpool2d_kernel(x, output, N_0, N_1, N_2, N_3):
    prickle.label('i')
    for i in range(N_1):
        prickle.label('j')
        for j in range(N_2):
            output[:,i,j,:] = -prickle.inf
            for a in range(N_0):
                for b in range(N_3):
                    for ii in range(2):
                        for jj in range(2):
                            output[a,i,j,b] = prickle.max(output[a,i,j,b], x[a,2*i+ii,2*j+jj,b])

# 2x2 maxpool operator, as used in LeNet-5
def maxpool2d(x):
    output = prickle.buffer.empty(
        (x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]),
        x.dtype, x.backend
    )
    N_0, N_1, N_2, N_3 = output.shape
    p = {'i': prickle.threads(N_1), 'j': prickle.threads(N_2)}
    maxpool2d_kernel(x, output, N_0, N_1, N_2, N_3, opts=prickle.par(p))
    return output

# LeNet-5 Convolutional Neural Network (inference mode)
def lenet5(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b,
           fc3w, fc3b, N, C_before_fc1):
    x = relu(conv2d(input, conv1, conv1bias))
    x = maxpool2d(x)
    x = relu(conv2d(x, conv2, conv2bias))
    x = maxpool2d(x)
    x = x.numpy()
    x = np.reshape(x, (N, C_before_fc1))
    x = relu(x @ fc1w + fc1b)
    x = relu(x @ fc2w + fc2b)
    return x @ fc3w + fc3b
