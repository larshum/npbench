# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------

import numpy as np
import torch


def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, itermax, horizon=2.0):
    # Adapted from
    # https://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/
    Xi, Yi = np.mgrid[0:xn, 0:yn]
    Xi = torch.tensor(Xi, device='cuda')
    Yi = torch.tensor(Yi, device='cuda')
    X = torch.linspace(xmin, xmax, xn, dtype=torch.float64, device=Xi.device)[Xi]
    Y = torch.linspace(ymin, ymax, yn, dtype=torch.float64, device=Yi.device)[Yi]
    C = X + Y * 1j
    N_ = torch.zeros(C.shape, dtype=torch.int64, device='cuda')
    Z_ = torch.zeros(C.shape, dtype=torch.complex128, device='cuda')
    Xi = Xi.reshape(xn * yn)
    Yi = Yi.reshape(xn * yn)
    C = C.reshape(xn * yn)

    Z = torch.zeros(C.shape, dtype=torch.complex128, device='cuda')
    for i in range(itermax):
        if not len(Z):
            break

        # Compute for relevant points only
        Z = torch.multiply(Z, Z)
        Z = torch.add(C, Z)

        # Failed convergence
        I = abs(Z) > horizon
        N_[Xi[I], Yi[I]] = i + 1
        Z_[Xi[I], Yi[I]] = Z[I]

        # Keep going with those who have not diverged yet
        I = torch.logical_not(I)
        Z = Z[I]
        Xi, Yi = Xi[I], Yi[I]
        C = C[I]
    return Z_.T, N_.T
