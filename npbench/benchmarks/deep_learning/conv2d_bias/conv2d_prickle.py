import prickle

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
def conv2d_bias(input, weights, bias):
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
