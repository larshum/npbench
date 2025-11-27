# Adapted from https://github.com/pmocz/nbody-python/blob/master/nbody.py
# TODO: Add GPL-3.0 License

import parpy
from parpy.math import sqrt
"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""


@parpy.jit
def getAcc_kernel(pos, mass, G, softening, dx, dy, dz, inv_r3, a, N):
    # matrix that stores all pairwise particle separations: r_j - r_i
    parpy.label('N2')
    for ij in range(N*N):
        i = ij // N
        j = ij % N
        dx[i,j] = pos[j,0] - pos[i,0]
        dy[i,j] = pos[j,1] - pos[i,1]
        dz[i,j] = pos[j,2] - pos[i,2]

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    parpy.label('N2')
    for ij in range(N*N):
        i = ij // N
        j = ij % N
        inv_r3[i,j] = dx[i,j]**2.0 + dy[i,j]**2.0 + dz[i,j]**2.0 + softening**2.0
        if inv_r3[i,j] > 0.0:
            inv_r3[i,j] **= (-1.5)

    # Compute and pack together the acceleration components in one go
    parpy.label('N')
    for i in range(0, N):
        parpy.label('reduce')
        a[i,0] = parpy.reduce.sum(G * (dx[i,:] * inv_r3[i,:]) * mass[:,0])
        parpy.label('reduce')
        a[i,1] = parpy.reduce.sum(G * (dy[i,:] * inv_r3[i,:]) * mass[:,0])
        parpy.label('reduce')
        a[i,2] = parpy.reduce.sum(G * (dz[i,:] * inv_r3[i,:]) * mass[:,0])

@parpy.jit
def getEnergy_kernel(pos, vel, mass, G, KE, PE, dx, dy, dz, inv_r, tmp, N):
    with parpy.gpu:
        parpy.label('reduce')
        KE[0] = parpy.reduce.sum(mass[:,0] * (vel[:,0]**2.0 + vel[:,1]**2.0 + vel[:,2]**2.0))
        KE[0] *= 0.5

    parpy.label('N2')
    for ij in range(N*N):
        i = ij // N
        j = ij % N
        dx[i,j] = pos[j,0] - pos[i,0]
        dy[i,j] = pos[j,1] - pos[i,1]
        dz[i,j] = pos[j,2] - pos[i,2]

    parpy.label('N2')
    for ij in range(N*N):
        i = ij // N
        j = ij % N
        inv_r[i,j] = sqrt(dx[i,j]**2.0 + dy[i,j]**2.0 + dz[i,j]**2.0)
        if inv_r[i,j] > 0.0:
            inv_r[i,j] = 1.0 / inv_r[i,j]

    parpy.label('N2')
    for ij in range(N*N):
        i = ij // N
        j = ij % N
        tmp[i,j] = -(mass[i,0] * mass[j,0]) * inv_r[i,j]

    with parpy.gpu:
        PE[0] = 0.0
        parpy.label('reduce')
        for ij in range(N*N):
            PE[0] += tmp[ij//N,ij%N] if ij%N < ij//N else 0.0
        PE[0] *= G

@parpy.jit
def nbody_kernel(mass, pos, vel, N, Nt, dt, G, softening, KE, PE, dx, dy, dz, acc, inv_r, tmp):
    parpy.builtin.inline(getAcc_kernel(pos, mass, G, softening, dx, dy, dz, inv_r, acc, N))
    parpy.builtin.inline(getEnergy_kernel(pos, vel, mass, G, KE[0], PE[0], dx, dy, dz, inv_r, tmp, N))
    for i in range(1, Nt+1):
        # (1/2) kick
        parpy.label('N')
        parpy.label('_')
        vel[:,:] += acc[:,:] * dt / 2.0

        # drift
        parpy.label('N')
        parpy.label('_')
        pos[:,:] += vel[:,:] * dt

        # update accelerations
        parpy.builtin.inline(getAcc_kernel(pos, mass, G, softening, dx, dy, dz, inv_r, acc, N))

        # (1/2) kick
        parpy.label('N')
        parpy.label('_')
        vel[:,:] += acc[:,:] * dt / 2.0

        # get energy of system
        parpy.builtin.inline(getEnergy_kernel(pos, vel, mass, G, KE[i], PE[i], dx, dy, dz, inv_r, tmp, N))

@parpy.jit
def nbody_center_of_mass_kernel(mass, vel, t1, t2, N):
    parpy.label('N')
    for i in range(N):
        t1[:] += mass[i,:] * vel[i,:]
        t2[:] += mass[i,:]

    parpy.label('N')
    for i in range(N):
        vel[i,:] -= t1[:] / t2[:]

def nbody(mass, pos, vel, N, Nt, dt, G, softening):
    # Convert to Center-of-Mass frame
    t1 = parpy.buffer.zeros((3,), vel.dtype, vel.backend())
    t2 = parpy.buffer.zeros((3,), vel.dtype, vel.backend())
    nbody_center_of_mass_kernel(mass, vel, t1, t2, N, opts=parpy.par({'N': parpy.threads(N)}))

    # Allocate temporary data used within the megakernel
    N,_ = pos.shape
    # NOTE: We add a dummy dimension to KE and PE to ensure they are passed as
    # tensors to the underlying kernels, so that we can modify individual
    # elements within kernels.
    KE = parpy.buffer.empty((Nt + 1, 1), pos.dtype, pos.backend())
    PE = parpy.buffer.empty_like(KE)
    a = parpy.buffer.empty((N, 3), pos.dtype, pos.backend())
    dx = parpy.buffer.empty((N, N), pos.dtype, pos.backend())
    dy = parpy.buffer.empty_like(dx)
    dz = parpy.buffer.empty_like(dx)
    inv_r = parpy.buffer.empty_like(dx)
    tmp = parpy.buffer.empty_like(dx)

    p = {
        'N2': parpy.threads(N*N),
        'N': parpy.threads(N),
        'reduce': parpy.threads(512).par_reduction()
    }
    nbody_kernel(
        mass, pos, vel, N, Nt, dt, G, softening, KE, PE, dx, dy, dz, a, inv_r, tmp,
        opts=parpy.par(p)
    )
    return KE.reshape(Nt+1), PE.reshape(Nt+1)
