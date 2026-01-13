import parpy
import torch


@parpy.jit
def conv2d_kernel(inputs, weights, output, H_out, W_out, N, C_in, C_out, K):
    parpy.label('H_out')
    for i in range(H_out):
        parpy.label('W_out')
        for j in range(W_out):
            output[:,i,j,:] = 0.0
            for a in range(N):
                for b in range(K):
                    for c in range(K):
                        for d in range(C_in):
                            parpy.label('C_out')
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

def conv2d_bias(input, weights, bias):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_in = input.shape[3]
    C_out = weights.shape[3]
    output = parpy.buffer.empty((N, H_out, W_out, C_out), parpy.types.F32, weights.backend())
    p = {
        'H_out': parpy.threads(H_out),
        'W_out': parpy.threads(W_out),
        'C_out': parpy.threads(C_out)
    }
    conv2d_kernel(input, weights, output, H_out, W_out, N, C_in, C_out, K, opts=parpy.par(p))
    p = {'i': parpy.threads(C_out), 'j': parpy.threads(N), 'k': parpy.threads(H_out), 'l': parpy.threads(W_out)}
    add_elemwise(output, bias, C_out, opts=parpy.par(p))
