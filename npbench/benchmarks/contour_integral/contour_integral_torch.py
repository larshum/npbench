# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import torch


def contour_integral(NR, NM, slab_per_bc, Ham, int_pts, Y):
    P0 = torch.zeros((NR, NM), dtype=torch.complex128, device=int_pts.device)
    P1 = torch.zeros((NR, NM), dtype=torch.complex128, device=int_pts.device)
    for z in int_pts:
        Tz = torch.zeros((NR, NR), dtype=torch.complex128, device=int_pts.device)
        for n in range(slab_per_bc + 1):
            zz = z ** (slab_per_bc / 2 - n)
            Tz += zz * Ham[n]
        if NR == NM:
            X = torch.linalg.inv(Tz)
        else:
            X = torch.linalg.solve(Tz, Y)
        if abs(z) < 1.0:
            X = -X
        P0 += X
        P1 += z * X

    return P0, P1
