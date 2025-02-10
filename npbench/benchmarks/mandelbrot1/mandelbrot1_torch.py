# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------

import torch


def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    # Adapted from https://www.ibm.com/developerworks/community/blogs/jfp/...
    #              .../entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en
    X = torch.linspace(xmin, xmax, xn, dtype=torch.float64, device='cuda')
    Y = torch.linspace(ymin, ymax, yn, dtype=torch.float64, device='cuda')
    C = X + Y[:, None] * 1j
    N = torch.zeros(C.shape, dtype=torch.int64, device='cuda')
    Z = torch.zeros(C.shape, dtype=torch.complex128, device='cuda')
    for n in range(maxiter):
        I = torch.less(abs(Z), horizon)
        N[I] = n
        Z[I] = Z[I]**2 + C[I]
    N[N == maxiter - 1] = 0
    return Z, N
