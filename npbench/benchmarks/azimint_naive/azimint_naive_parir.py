# Copyright 2014 Jérôme Kieffer et al.
# This is an open-access article distributed under the terms of the
# Creative Commons Attribution License, which permits unrestricted use,
# distribution, and reproduction in any medium, provided the original author
# and source are credited.
# http://creativecommons.org/licenses/by/3.0/
# Jérôme Kieffer and Giannis Ashiotis. Pyfai: a python library for
# high performance azimuthal integration on gpu, 2014. In Proceedings of the
# 7th European Conference on Python in Science (EuroSciPy 2014).

import parir
import torch

@parir.jit
def parir_kernel(data, radius, res, rmax, npt, N):
    for ix in range(N):
        rmax[0] = max(rmax[0], radius[ix])
    for i in range(npt):
        r1 = rmax[0] * parir.float64(i) / parir.float64(npt)
        r2 = rmax[0] * parir.float64(i+1) / parir.float64(npt)
        c = 0.0
        for j in range(N):
            c = c + (1.0 if r1 <= radius[j] and radius[j] < r2 else 0.0)
        s = 0.0
        for j in range(N):
            s = s + (1.0 if r1 <= radius[j] and radius[j] < r2 else 0.0) * data[j]
        res[i] = s / c

def azimint_naive(data, radius, npt):
    N, = data.shape
    rmax = torch.empty((1,), dtype=torch.float64, device=data.device)
    res = torch.zeros(npt, dtype=torch.float64, device=data.device)
    p = {
        'i': [parir.threads(npt)],
        'ix': [parir.threads(1024), parir.reduce()],
        'j': [parir.threads(1024), parir.reduce()]
    }
    parir_kernel(data, radius, res, rmax, npt, N, parallelize=p)
    return res
