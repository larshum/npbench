# Copyright 2014 Jérôme Kieffer et al.
# This is an open-access article distributed under the terms of the
# Creative Commons Attribution License, which permits unrestricted use,
# distribution, and reproduction in any medium, provided the original author
# and source are credited.
# http://creativecommons.org/licenses/by/3.0/
# Jérôme Kieffer and Giannis Ashiotis. Pyfai: a python library for
# high performance azimuthal integration on gpu, 2014. In Proceedings of the
# 7th European Conference on Python in Science (EuroSciPy 2014).

import parpy
from parpy.builtin import convert
from parpy.types import F64

T = parpy.types.type_var()
N = parpy.types.shape_var()

@parpy.jit
def parpy_kernel(
        data: parpy.types.buffer(T, [N]),
        radius: parpy.types.buffer(T, [N]),
        res,
        rmax,
        npt
):
    for i in range(N):
        rmax[0] = parpy.builtin.maximum(rmax[0], radius[i])
    parpy.label('npt')
    for i in range(npt):
        r1 = rmax[0] * convert(i, F64) / convert(npt, F64)
        r2 = rmax[0] * convert(i+1, F64) / convert(npt, F64)
        c = parpy.reduce.sum(1.0 if r1 <= radius[0:N] and radius[0:N] < r2 else 0.0)
        s = parpy.reduce.sum((1.0 if r1 <= radius[0:N] and radius[0:N] < r2 else 0.0) * data[0:N])
        res[i] = s / c

def azimint_naive(data, radius, npt):
    N, = data.shape
    rmax = parpy.buffer.empty((1,), data.dtype, data.backend())
    res = parpy.buffer.zeros((npt,), data.dtype, data.backend())
    p = {
        'npt': parpy.threads(npt),
        'N': parpy.threads(1024).par_reduction(),
    }
    opts = parpy.par(p)
    opts.max_unroll_count = 0
    parpy_kernel(data, radius, res, rmax, npt, opts=opts)
    return res
