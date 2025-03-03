# Adapted from https://github.com/pmocz/nbody-python/blob/master/nbody.py
# TODO: Add GPL-3.0 License

import parir
import torch
"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""


@parir.jit
def getAcc_kernel(pos, mass, G, softening, dx, dy, dz, inv_r3, a, N):
    # matrix that stores all pairwise particle separations: r_j - r_i
    parir.label('N2')
    for ij in range(N*N):
        i = ij // N
        j = ij % N
        dx[i,j] = pos[j,0] - pos[i,0]
        dy[i,j] = pos[j,1] - pos[i,1]
        dz[i,j] = pos[j,2] - pos[i,2]

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    parir.label('N2')
    for ij in range(N*N):
        i = ij // N
        j = ij % N
        inv_r3[i,j] = dx[i,j]**2.0 + dy[i,j]**2.0 + dz[i,j]**2.0 + softening**2.0
        if inv_r3[i,j] > 0.0:
            inv_r3[i,j] **= (-1.5)

    # Compute and pack together the acceleration components in one go
    parir.label('N')
    for i in range(0, N):
        parir.label('reduce')
        a[i,0] = parir.sum(G * (dx[i,:] * inv_r3[i,:]) * mass[:,0])
        parir.label('reduce')
        a[i,1] = parir.sum(G * (dy[i,:] * inv_r3[i,:]) * mass[:,0])
        parir.label('reduce')
        a[i,2] = parir.sum(G * (dz[i,:] * inv_r3[i,:]) * mass[:,0])

@parir.jit
def getEnergy_kernel(pos, vel, mass, G, KE, PE, dx, dy, dz, inv_r, tmp, N):
    with parir.gpu:
        parir.label('i')
        KE[0] = parir.sum(mass[:,0] * (vel[:,0]**2.0 + vel[:,1]**2.0 + vel[:,2]**2.0))
        KE[0] *= 0.5

    parir.label('N2')
    for ij in range(N*N):
        i = ij // N
        j = ij % N
        dx[i,j] = pos[j,0] - pos[i,0]
        dy[i,j] = pos[j,1] - pos[i,1]
        dz[i,j] = pos[j,2] - pos[i,2]

    parir.label('N2')
    for ij in range(N*N):
        i = ij // N
        j = ij % N
        inv_r[i,j] = parir.sqrt(dx[i,j]**2.0 + dy[i,j]**2.0 + dz[i,j]**2.0)
        if inv_r[i,j] > 0.0:
            inv_r[i,j] = 1.0 / inv_r[i,j]

    parir.label('N2')
    for ij in range(N*N):
        i = ij // N
        j = ij % N
        tmp[i,j] = -(mass[i,0] * mass[j,0]) * inv_r[i,j]

    with parir.gpu:
        PE[0] = 0.0
        for i in range(N):
            parir.label('reduce')
            for j in range(i+1, N):
                PE[0] += tmp[i,j]
        PE[0] *= G

@parir.jit
def nbody_kernel(mass, pos, vel, N, Nt, dt, G, softening, KE, PE, dx, dy, dz, acc, inv_r, tmp):
    getAcc_kernel(pos, mass, G, softening, dx, dy, dz, inv_r, acc, N)
    getEnergy_kernel(pos, vel, mass, G, KE[0], PE[0], dx, dy, dz, inv_r, tmp, N)
    for i in range(1, Nt+1):
        # (1/2) kick
        parir.label('N')
        parir.label('_')
        vel[:,:] += acc[:,:] * dt / 2.0

        # drift
        parir.label('N')
        parir.label('_')
        pos[:,:] += vel[:,:] * dt

        # update accelerations
        getAcc_kernel(pos, mass, G, softening, dx, dy, dz, inv_r, acc, N)

        # (1/2) kick
        parir.label('N')
        parir.label('_')
        vel[:,:] += acc[:,:] * dt / 2.0

        # get energy of system
        getEnergy_kernel(pos, vel, mass, G, KE[i], PE[i], dx, dy, dz, inv_r, tmp, N)

def nbody(mass, pos, vel, N, Nt, dt, G, softening):
    # Convert to Center-of-Mass frame
    vel -= torch.mean(mass * vel, axis=0) / torch.mean(mass)

    # Allocate temporary data used within the megakernel
    N,_ = pos.shape
    # NOTE: We add a dummy dimension to KE and PE to ensure they are passed as
    # tensors to the underlying kernels, so that we can modify individual
    # elements within kernels.
    KE = torch.empty((Nt + 1, 1), dtype=torch.float64, device=pos.device)
    PE = torch.empty_like(KE)
    a = torch.empty((N, 3), dtype=pos.dtype, device=pos.device)
    dx = torch.empty((N, N), dtype=pos.dtype, device=pos.device)
    dy = torch.empty_like(dx)
    dz = torch.empty_like(dx)
    inv_r = torch.empty_like(dx)
    tmp = torch.empty_like(dx)

    p = {
        'N2': [parir.threads(N*N)],
        'N': [parir.threads(N)],
        'reduce': [parir.threads(64), parir.reduce()]
    }
    nbody_kernel(mass, pos, vel, N, Nt, dt, G, softening, KE, PE, dx, dy, dz, a, inv_r, tmp, parallelize=p)
    return KE.reshape(Nt+1), PE.reshape(Nt+1)
