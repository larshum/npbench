# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------

import parir
from parir import ParKind
import torch

@parir.jit
def parir_kernel(N, Z, C, I, M, K, horizon, maxiter):
    for n in range(maxiter):
        for i in range(M):
            for j in range(K):
                I[i,j] = parir.sqrt(Z[i,j,0]**2.0+Z[i,j,1]**2.0) < horizon
        for i in range(M):
            for j in range(K):
                if I[i,j]:
                    N[i,j] = n
            for j in range(K):
                if I[i,j]:
                    tmp = Z[i,j,0]
                    Z[i,j,0] = Z[i,j,0]**2.0 - Z[i,j,1]**2.0 + C[i,j,0]
                    Z[i,j,1] = 2.0 * tmp * Z[i,j,1] + C[i,j,1]
    for i in range(M):
        for j in range(K):
            if N[i,j] == maxiter - 1:
                N[i,j] = 0

def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    X = torch.linspace(xmin, xmax, xn, dtype=torch.float64, device='cuda')
    Y = torch.linspace(ymin, ymax, yn, dtype=torch.float64, device='cuda')
    C = X + Y[:, None] * 1j
    N = torch.zeros(C.shape, dtype=torch.int64, device='cuda')
    Z = torch.zeros(C.shape, dtype=torch.complex128, device='cuda')
    I = torch.empty_like(Z, dtype=torch.bool)
    M, K = C.shape
    p = { 'i': [ParKind.GpuThreads(M)], 'j': [ParKind.GpuThreads(K)] }
    Z = torch.view_as_real(Z)
    C = torch.view_as_real(C)
    parir_kernel(N, Z, C, I, M, K, horizon, maxiter, parallelize=p)
    return torch.view_as_complex(Z), N
