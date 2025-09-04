# Barba, Lorena A., and Forsyth, Gilbert F. (2018).
# CFD Python: the 12 steps to Navier-Stokes equations.
# Journal of Open Source Education, 1(9), 21,
# https://doi.org/10.21105/jose.00021
# TODO: License
# (c) 2017 Lorena A. Barba, Gilbert F. Forsyth.
# All content is under Creative Commons Attribution CC-BY 4.0,
# and all code is under BSD-3 clause (previously under MIT, and changed on March 8, 2018).

import parpy
import torch


@parpy.jit
def build_up_b(rho, dt, dx, dy, u, v, b):
    parpy.label('ny')
    parpy.label('nx')
    b[1:-1,
      1:-1] = (rho * (1.0 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2.0 * dx) +
                                (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2.0 * dy)) -
                      ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2.0 * dx))**2.0 - 2.0 *
                      ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2.0 * dy) *
                       (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2.0 * dx)) -
                      ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2.0 * dy))**2.0))

    # Periodic BC Pressure @ x = 2
    parpy.label('ny')
    b[1:-1, -1] = (rho * (1.0 / dt * ((u[1:-1, 0] - u[1:-1, -2]) / (2.0 * dx) +
                                    (v[2:, -1] - v[0:-2, -1]) / (2.0 * dy)) -
                          ((u[1:-1, 0] - u[1:-1, -2]) / (2.0 * dx))**2.0 - 2.0 *
                          ((u[2:, -1] - u[0:-2, -1]) / (2.0 * dy) *
                           (v[1:-1, 0] - v[1:-1, -2]) / (2.0 * dx)) -
                          ((v[2:, -1] - v[0:-2, -1]) / (2.0 * dy))**2.0))

    # Periodic BC Pressure @ x = 0
    parpy.label('ny')
    b[1:-1, 0] = (rho * (1.0 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2.0 * dx) +
                                   (v[2:, 0] - v[0:-2, 0]) / (2.0 * dy)) -
                         ((u[1:-1, 1] - u[1:-1, -1]) / (2.0 * dx))**2.0 - 2.0 *
                         ((u[2:, 0] - u[0:-2, 0]) / (2.0 * dy) *
                          (v[1:-1, 1] - v[1:-1, -1]) /
                          (2.0 * dx)) - ((v[2:, 0] - v[0:-2, 0]) / (2.0 * dy))**2.0))


@parpy.jit
def pressure_poisson_periodic(nit, p, dx, dy, b, pn):
    for q in range(nit):
        parpy.label('ny')
        parpy.label('nx')
        pn[:,:] = p[:,:]

        parpy.label('ny')
        parpy.label('nx')
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2.0 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2.0) /
                         (2.0 * (dx**2.0 + dy**2.0)) - dx**2.0 * dy**2.0 /
                         (2.0 * (dx**2.0 + dy**2.0)) * b[1:-1, 1:-1])

        # Periodic BC Pressure @ x = 2
        parpy.label('ny')
        p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2]) * dy**2.0 +
                        (pn[2:, -1] + pn[0:-2, -1]) * dx**2.0) /
                       (2.0 * (dx**2.0 + dy**2.0)) - dx**2.0 * dy**2.0 /
                       (2.0 * (dx**2.0 + dy**2.0)) * b[1:-1, -1])

        # Periodic BC Pressure @ x = 0
        parpy.label('ny')
        p[1:-1,
          0] = (((pn[1:-1, 1] + pn[1:-1, -1]) * dy**2.0 +
                 (pn[2:, 0] + pn[0:-2, 0]) * dx**2.0) / (2.0 * (dx**2.0 + dy**2.0)) -
                dx**2.0 * dy**2.0 / (2.0 * (dx**2.0 + dy**2.0)) * b[1:-1, 0])

        # Wall boundary conditions, pressure
        parpy.label('nx')
        p[-1, :] = p[-2, :]  # dp/dy = 0 at y = 2
        parpy.label('nx')
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0


@parpy.jit
def channel_flow_kernel(nit, u, v, dt, dx, dy, p, rho, nu, F, un, vn, pn, b, udiff):
    # Copy u -> un and v -> vn
    parpy.label('ny')
    parpy.label('nx')
    un[:,:] = u[:,:]
    parpy.label('ny')
    parpy.label('nx')
    vn[:,:] = v[:,:]

    build_up_b(rho, dt, dx, dy, u, v, b)
    pressure_poisson_periodic(nit, p, dx, dy, b, pn)

    parpy.label('ny')
    parpy.label('nx')
    u[1:-1,
      1:-1] = (un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx *
               (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
               vn[1:-1, 1:-1] * dt / dy *
               (un[1:-1, 1:-1] - un[0:-2, 1:-1]) - dt / (2.0 * rho * dx) *
               (p[1:-1, 2:] - p[1:-1, 0:-2]) + nu *
               (dt / dx**2.0 *
                (un[1:-1, 2:] - 2.0 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                dt / dy**2.0 *
                (un[2:, 1:-1] - 2.0 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) +
               F * dt)

    parpy.label('ny')
    parpy.label('nx')
    v[1:-1,
      1:-1] = (vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx *
               (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
               vn[1:-1, 1:-1] * dt / dy *
               (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - dt / (2.0 * rho * dy) *
               (p[2:, 1:-1] - p[0:-2, 1:-1]) + nu *
               (dt / dx**2.0 *
                (vn[1:-1, 2:] - 2.0 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                dt / dy**2.0 *
                (vn[2:, 1:-1] - 2.0 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

    # Periodic BC u @ x = 2
    parpy.label('ny')
    u[1:-1, -1] = (
        un[1:-1, -1] - un[1:-1, -1] * dt / dx *
        (un[1:-1, -1] - un[1:-1, -2]) - vn[1:-1, -1] * dt / dy *
        (un[1:-1, -1] - un[0:-2, -1]) - dt / (2.0 * rho * dx) *
        (p[1:-1, 0] - p[1:-1, -2]) + nu *
        (dt / dx**2.0 *
         (un[1:-1, 0] - 2.0 * un[1:-1, -1] + un[1:-1, -2]) + dt / dy**2.0 *
         (un[2:, -1] - 2.0 * un[1:-1, -1] + un[0:-2, -1])) + F * dt)

    # Periodic BC u @ x = 0
    parpy.label('ny')
    u[1:-1,
      0] = (un[1:-1, 0] - un[1:-1, 0] * dt / dx *
            (un[1:-1, 0] - un[1:-1, -1]) - vn[1:-1, 0] * dt / dy *
            (un[1:-1, 0] - un[0:-2, 0]) - dt / (2.0 * rho * dx) *
            (p[1:-1, 1] - p[1:-1, -1]) + nu *
            (dt / dx**2.0 *
             (un[1:-1, 1] - 2.0 * un[1:-1, 0] + un[1:-1, -1]) + dt / dy**2.0 *
             (un[2:, 0] - 2.0 * un[1:-1, 0] + un[0:-2, 0])) + F * dt)

    # Periodic BC v @ x = 2
    parpy.label('ny')
    v[1:-1, -1] = (
        vn[1:-1, -1] - un[1:-1, -1] * dt / dx *
        (vn[1:-1, -1] - vn[1:-1, -2]) - vn[1:-1, -1] * dt / dy *
        (vn[1:-1, -1] - vn[0:-2, -1]) - dt / (2.0 * rho * dy) *
        (p[2:, -1] - p[0:-2, -1]) + nu *
        (dt / dx**2.0 *
         (vn[1:-1, 0] - 2.0 * vn[1:-1, -1] + vn[1:-1, -2]) + dt / dy**2.0 *
         (vn[2:, -1] - 2.0 * vn[1:-1, -1] + vn[0:-2, -1])))

    # Periodic BC v @ x = 0
    parpy.label('ny')
    v[1:-1,
      0] = (vn[1:-1, 0] - un[1:-1, 0] * dt / dx *
            (vn[1:-1, 0] - vn[1:-1, -1]) - vn[1:-1, 0] * dt / dy *
            (vn[1:-1, 0] - vn[0:-2, 0]) - dt / (2.0 * rho * dy) *
            (p[2:, 0] - p[0:-2, 0]) + nu *
            (dt / dx**2.0 *
             (vn[1:-1, 1] - 2.0 * vn[1:-1, 0] + vn[1:-1, -1]) + dt / dy**2.0 *
             (vn[2:, 0] - 2.0 * vn[1:-1, 0] + vn[0:-2, 0])))

    # Wall BC: u,v = 0 @ y = 0,2
    parpy.label('nx')
    u[0, :] = 0.0
    parpy.label('nx')
    u[-1, :] = 0.0
    parpy.label('nx')
    v[0, :] = 0.0
    parpy.label('nx')
    v[-1, :] = 0.0

    # Compute udiff = (sum(u) - sum(un)) / sum(u)
    parpy.label('reduce')
    udiff[1] = parpy.operators.sum(u[:,:])
    parpy.label('reduce')
    udiff[2] = parpy.operators.sum(un[:,:])
    with parpy.gpu:
        udiff[0] = (udiff[1] - udiff[2]) / udiff[1]

def channel_flow(nit, u, v, dt, dx, dy, p, rho, nu, F):
    un = parpy.buffer.empty_like(u)
    vn = parpy.buffer.empty_like(v)
    pn = parpy.buffer.empty_like(p)
    b = parpy.buffer.zeros_like(u)
    udiff_tmp = parpy.buffer.empty((3,), u.dtype, u.backend)

    udiff = 1
    stepcount = 0

    ny, nx = u.shape
    while udiff > .001:
        par = {
            'ny': parpy.threads(ny),
            'nx': parpy.threads(nx),
            'reduce': parpy.threads(1024),
        }
        channel_flow_kernel(
            nit, u, v, dt, dx, dy, p, rho, nu, F, un, vn, pn, b, udiff_tmp,
            opts=parpy.par(par)
        )
        udiff = udiff_tmp.numpy()[0]
        stepcount += 1

    return stepcount
