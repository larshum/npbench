# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------

import prickle
import torch

@prickle.jit
def prickle_kernel(N, Z, C, I, M, K, horizon, maxiter):
    for n in range(maxiter):
        prickle.label('i')
        prickle.label('j')
        I[:,:] = prickle.sqrt(Z[:,:,0]**2.0 + Z[:,:,1]**2.0) < horizon
        prickle.label('i')
        for i in range(M):
            prickle.label('j')
            for j in range(K):
                if I[i,j]:
                    N[i,j] = n
            prickle.label('j')
            for j in range(K):
                if I[i,j]:
                    tmp = Z[i,j,0]
                    Z[i,j,0] = Z[i,j,0]**2.0 - Z[i,j,1]**2.0 + C[i,j,0]
                    Z[i,j,1] = 2.0 * tmp * Z[i,j,1] + C[i,j,1]
    prickle.label('i')
    for i in range(M):
        prickle.label('j')
        for j in range(K):
            if N[i,j] == maxiter - 1:
                N[i,j] = 0

def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    X = torch.linspace(xmin, xmax, xn, dtype=torch.float64)
    Y = torch.linspace(ymin, ymax, yn, dtype=torch.float64)
    C = X + Y[:, None] * 1j
    N = torch.zeros(C.shape, dtype=torch.int64)
    Z = torch.zeros(C.shape, dtype=torch.complex128)
    I = torch.empty_like(Z, dtype=torch.bool)
    M, K = C.shape
    p = { 'i': prickle.threads(M), 'j': prickle.threads(K) }
    Z = torch.view_as_real(Z)
    C = torch.view_as_real(C)
    prickle_kernel(N, Z, C, I, M, K, horizon, maxiter, opts=prickle.par(p))
    return torch.view_as_complex(Z), N
