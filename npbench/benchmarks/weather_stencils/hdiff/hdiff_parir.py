import parir
import torch


@parir.jit
def hdiff_kernel(in_field, out_field, coeff, lap_field, res1, res2, flx_field, fly_field, I, J, K):
    parir.label('I')
    parir.label('J')
    parir.label('K')
    lap_field[:,:,:] = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (
        in_field[2:I + 4, 1:J + 3, :] + in_field[0:I + 2, 1:J + 3, :] +
        in_field[1:I + 3, 2:J + 4, :] + in_field[1:I + 3, 0:J + 2, :])

    parir.label('I')
    parir.label('J')
    parir.label('K')
    res1[:,:,:] = lap_field[1:, 1:J + 1, :] - lap_field[:-1, 1:J + 1, :]

    parir.label('I')
    parir.label('J')
    parir.label('K')
    flx_field[:,:,:] = (0.0
            if res1[:,:,:] * (in_field[2:I+3,2:J+2,:] - in_field[1:I+2,2:J+2,:]) > 0.0
            else res1[:,:,:])

    parir.label('I')
    parir.label('J')
    parir.label('K')
    res2[:,:,:] = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :-1, :]

    parir.label('I')
    parir.label('J')
    parir.label('K')
    fly_field[:,:,:] = (0.0
            if res2[:,:,:] * (in_field[2:I+2,2:J+3,:] - in_field[2:I+2,1:J+2,:]) > 0.0
            else res2[:,:,:])

    parir.label('I')
    parir.label('J')
    parir.label('K')
    out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (
        flx_field[1:, :, :] - flx_field[:-1, :, :] + fly_field[:, 1:, :] -
        fly_field[:, :-1, :])


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L194
def hdiff(in_field, out_field, coeff):
    I, J, K = out_field.shape[0], out_field.shape[1], out_field.shape[2]
    lap_field = torch.empty(I+2,J+2,K, dtype=in_field.dtype, device=in_field.device)
    res1 = torch.empty(I+1,J,K, dtype=in_field.dtype, device=in_field.device)
    res2 = torch.empty(I,J+1,K, dtype=in_field.dtype, device=in_field.device)
    flx_field = torch.empty(I+1,J,K, dtype=in_field.dtype, device=in_field.device)
    fly_field = torch.empty(I,J+1,K, dtype=in_field.dtype, device=in_field.device)
    p = {'I': parir.threads(I), 'J': parir.threads(J), 'K': parir.threads(K)}
    hdiff_kernel(
        in_field, out_field, coeff, lap_field, res1, res2, flx_field, fly_field, I, J, K,
        opts=parir.par(p)
    )
