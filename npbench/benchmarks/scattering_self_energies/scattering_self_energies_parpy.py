# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import parpy
from parpy.operators import sum

@parpy.jit
def scattering_self_energies_kernel(neigh_idx, dH, G, D, Sigma, dHG, dHD, Nkz,
                                    NE, Nqz, Nw, N3D, NA, NB, Norb):
    parpy.label('Nkz')
    for k in range(Nkz):
        parpy.label('NE')
        for E in range(NE):
            for q in range(Nqz):
                for w in range(Nw):
                    for i in range(N3D):
                        for j in range(N3D):
                            parpy.label('NA')
                            for a in range(NA):
                                for b in range(NB):
                                    if E - w >= 0:
                                        idx = neigh_idx[a, b]
                                        parpy.label('threads')
                                        for xyz in range(Norb * Norb * 2):
                                            x = xyz // (Norb * 2)
                                            y = (xyz // 2) % Norb
                                            z = xyz % 2
                                            z_inv = (z + 1) % 2
                                            sign = -1.0 if z == 0 else 1.0

                                            # first complex row/column matrix
                                            # multiplication
                                            t1 = sum(G[k,E-w,idx,x,:,0] * dH[a,b,i,:,y,z])
                                            t2 = sum(G[k,E-w,idx,x,:,1] * dH[a,b,i,:,y,z_inv])
                                            dHG[k,E,a,x,y,z] = t1 + t2 * sign

                                            # complex elementwise multiplication
                                            dHD[k,E,a,x,y,z] = \
                                                dH[a,b,j,x,y,0] * D[q,w,a,b,i,j,z] + \
                                                dH[a,b,j,x,y,1] * D[q,w,a,b,i,j,z_inv] * sign
                                        parpy.label('threads')
                                        for xyz in range(Norb * Norb * 2):
                                            x = xyz // (Norb * 2)
                                            y = (xyz // 2) % Norb
                                            z = xyz % 2
                                            z_inv = (z + 1) % 2
                                            sign = -1.0 if z == 0 else 1.0

                                            # second complex matrix multiply
                                            t1 = sum(dHG[k,E,a,x,:,0] * dHD[k,E,a,:,y,z])
                                            t2 = sum(dHG[k,E,a,x,:,1] * dHD[k,E,a,:,y,z_inv])
                                            Sigma[k,E,a,x,y,z] += t1 + t2 * sign
                                    else:
                                        # Dummy else-branch to balance
                                        # parallelism across both branches.
                                        parpy.label('threads')
                                        for i in range(1):
                                            x = 0.0

def scattering_self_energies(neigh_idx, dH, G, D, Sigma):
    NA, NB = neigh_idx.shape
    Nkz, NE, NA, Norb, Norb, _ = G.shape
    Nqz, Nw, NA, NB, N3D, N3D, _ = D.shape
    p = {
        'Nkz': parpy.threads(Nkz),
        'NE': parpy.threads(NE),
        'NA': parpy.threads(NA),
        'threads': parpy.threads(32)
    }
    dHG = parpy.buffer.zeros((Nkz, NE, NA, Norb, Norb, 2), G.dtype, G.backend)
    dHD = parpy.buffer.zeros_like(dHG)
    scattering_self_energies_kernel(
        neigh_idx, dH, G, D, Sigma, dHG, dHD,
        Nkz, NE, Nqz, Nw, N3D, NA, NB, Norb, opts=parpy.par(p)
    )
