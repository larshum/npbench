# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import prickle
import torch

@prickle.jit
def scattering_self_energies_kernel(neigh_idx, dH, G, D, Sigma, dHG, dHD, Nkz,
                                    NE, Nqz, Nw, N3D, NA, NB, Norb):
    prickle.label('Nkz')
    for k in range(Nkz):
        prickle.label('NE')
        for E in range(NE):
            for q in range(Nqz):
                for w in range(Nw):
                    for i in range(N3D):
                        for j in range(N3D):
                            prickle.label('NA')
                            for a in range(NA):
                                for b in range(NB):
                                    if E - w >= 0:
                                        idx = neigh_idx[a, b]
                                        prickle.label('threads')
                                        for xyz in range(Norb * Norb * 2):
                                            x = xyz // (Norb * 2)
                                            y = (xyz // 2) % Norb
                                            z = xyz % 2
                                            z_inv = (z + 1) % 2
                                            sign = -1.0 if z == 0 else 1.0

                                            # first complex row/column matrix
                                            # multiplication
                                            t1 = prickle.sum(G[k,E-w,idx,x,:,0] * dH[a,b,i,:,y,z])
                                            t2 = prickle.sum(G[k,E-w,idx,x,:,1] * dH[a,b,i,:,y,z_inv])
                                            dHG[k,E,a,x,y,z] = t1 + t2 * sign

                                            # complex elementwise multiplication
                                            dHD[k,E,a,x,y,z] = \
                                                dH[a,b,j,x,y,0] * D[q,w,a,b,i,j,z] + \
                                                dH[a,b,j,x,y,1] * D[q,w,a,b,i,j,z_inv] * sign
                                        prickle.label('threads')
                                        for xyz in range(Norb * Norb * 2):
                                            x = xyz // (Norb * 2)
                                            y = (xyz // 2) % Norb
                                            z = xyz % 2
                                            z_inv = (z + 1) % 2
                                            sign = -1.0 if z == 0 else 1.0

                                            # second complex matrix multiply
                                            t1 = prickle.sum(dHG[k,E,a,x,:,0] * dHD[k,E,a,:,y,z])
                                            t2 = prickle.sum(dHG[k,E,a,x,:,1] * dHD[k,E,a,:,y,z_inv])
                                            Sigma[k,E,a,x,y,z] += t1 + t2 * sign
                                    else:
                                        # Dummy else-branch to balance
                                        # parallelism across both branches.
                                        prickle.label('threads')
                                        for i in range(1):
                                            x = 0.0

def scattering_self_energies(neigh_idx, dH, G, D, Sigma):
    NA, NB = neigh_idx.shape
    Nkz, NE, NA, Norb, Norb = G.shape
    Nqz, Nw, NA, NB, N3D, N3D = D.shape
    p = {
        'Nkz': prickle.threads(Nkz),
        'NE': prickle.threads(NE),
        'NA': prickle.threads(NA),
        'threads': prickle.threads(32)
    }
    dHG = torch.zeros(Nkz, NE, NA, Norb, Norb, dtype=G.dtype, device=G.device)
    dHD = torch.zeros_like(dHG)
    scattering_self_energies_kernel(
        neigh_idx, torch.view_as_real(dH), torch.view_as_real(G),
        torch.view_as_real(D), torch.view_as_real(Sigma),
        torch.view_as_real(dHG), torch.view_as_real(dHD),
        Nkz, NE, Nqz, Nw, N3D, NA, NB, Norb, opts=prickle.par(p)
    )
