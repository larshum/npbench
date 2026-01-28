import parpy

T = parpy.types.type_var()
I = parpy.types.shape_var()
J = parpy.types.shape_var()
K = parpy.types.shape_var()

@parpy.jit
def hdiff_kernel(
        in_field,
        out_field: parpy.types.buffer(T, [I, J, K]),
        coeff,
        lap_field,
        res1,
        res2,
        flx_field,
        fly_field):
    lap_field[:I+2,:J+2,:K] = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (
        in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] +
        in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])

    res1[:I+1,:J,:K] = lap_field[1:, 1:J + 1, :] - lap_field[:-1, 1:J + 1, :]

    flx_field[:I+1,:J,:K] = (0.0
            if res1[:,:,:] * (in_field[2:I+3,2:J+2,:] - in_field[1:I+2,2:J+2,:]) > 0.0
            else res1[:,:,:])

    res2[:I,:J+1,:K] = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :-1, :]

    fly_field[:I,:J+1,:K] = (0.0
            if res2[:,:,:] * (in_field[2:I+2,2:J+3,:] - in_field[2:I+2,1:J+2,:]) > 0.0
            else res2[:,:,:])

    out_field[:I, :J, :K] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (
        flx_field[1:, :, :] - flx_field[:-1, :, :] + fly_field[:, 1:, :] -
        fly_field[:, :-1, :])


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L194
def hdiff(in_field, out_field, coeff):
    I, J, K = out_field.shape
    lap_field = parpy.buffer.empty((I+2,J+2,K), in_field.dtype, in_field.backend())
    res1 = parpy.buffer.empty((I+1,J,K), in_field.dtype, in_field.backend())
    res2 = parpy.buffer.empty((I,J+1,K), in_field.dtype, in_field.backend())
    flx_field = parpy.buffer.empty((I+1,J,K), in_field.dtype, in_field.backend())
    fly_field = parpy.buffer.empty((I,J+1,K), in_field.dtype, in_field.backend())
    p = {'I': parpy.threads(I), 'J': parpy.threads(J), 'K': parpy.threads(K)}
    opts = parpy.par(p)
    opts.max_unroll_count = 0
    hdiff_kernel(
        in_field, out_field, coeff, lap_field, res1, res2, flx_field, fly_field,
        opts=opts
    )
