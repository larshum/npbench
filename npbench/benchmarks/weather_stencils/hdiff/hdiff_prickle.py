import prickle

@prickle.jit
def hdiff_kernel(in_field, out_field, coeff, lap_field, res1, res2, flx_field, fly_field, I, J, K):
    prickle.label('I')
    prickle.label('J')
    prickle.label('K')
    lap_field[:,:,:] = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (
        in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] +
        in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])

    prickle.label('I')
    prickle.label('J')
    prickle.label('K')
    res1[:,:,:] = lap_field[1:, 1:J + 1, :] - lap_field[:-1, 1:J + 1, :]

    prickle.label('I')
    prickle.label('J')
    prickle.label('K')
    flx_field[:,:,:] = (0.0
            if res1[:,:,:] * (in_field[2:I+3,2:J+2,:] - in_field[1:I+2,2:J+2,:]) > 0.0
            else res1[:,:,:])

    prickle.label('I')
    prickle.label('J')
    prickle.label('K')
    res2[:,:,:] = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :-1, :]

    prickle.label('I')
    prickle.label('J')
    prickle.label('K')
    fly_field[:,:,:] = (0.0
            if res2[:,:,:] * (in_field[2:I+2,2:J+3,:] - in_field[2:I+2,1:J+2,:]) > 0.0
            else res2[:,:,:])

    prickle.label('I')
    prickle.label('J')
    prickle.label('K')
    out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (
        flx_field[1:, :, :] - flx_field[:-1, :, :] + fly_field[:, 1:, :] -
        fly_field[:, :-1, :])


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L194
def hdiff(in_field, out_field, coeff):
    I, J, K = out_field.shape[0], out_field.shape[1], out_field.shape[2]
    lap_field = prickle.buffer.empty((I+2,J+2,K), in_field.dtype, in_field.backend)
    res1 = prickle.buffer.empty((I+1,J,K), in_field.dtype, in_field.backend)
    res2 = prickle.buffer.empty((I,J+1,K), in_field.dtype, in_field.backend)
    flx_field = prickle.buffer.empty((I+1,J,K), in_field.dtype, in_field.backend)
    fly_field = prickle.buffer.empty((I,J+1,K), in_field.dtype, in_field.backend)
    p = {'I': prickle.threads(I), 'J': prickle.threads(J), 'K': prickle.threads(K)}
    hdiff_kernel(
        in_field, out_field, coeff, lap_field, res1, res2, flx_field, fly_field, I, J, K,
        opts=prickle.par(p)
    )
