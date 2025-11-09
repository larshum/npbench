import numpy as np
import torch


def stockham_fft(N, R, K, x, y):

    # Generate DFT matrix for radix R.
    # Define transient variable for matrix.
    i_coord, j_coord = np.mgrid[0:R, 0:R]
    i_coord = torch.tensor(i_coord, device=x.device)
    j_coord = torch.tensor(j_coord, device=x.device)
    dft_mat = torch.empty((R, R), dtype=torch.complex128, device=x.device)
    dft_mat = torch.exp(-2.0j * torch.pi * i_coord * j_coord / R).to(dtype=torch.complex128)
    # Move input x to output y
    # to avoid overwriting the input.
    y[:] = x[:]

    ii_coord, jj_coord = np.mgrid[0:R, 0:R**K]
    ii_coord = torch.tensor(ii_coord, device=x.device)
    jj_coord = torch.tensor(jj_coord, device=x.device)

    # Main Stockham loop
    for i in range(K):

        # Stride permutation
        yv = torch.reshape(y, (R**i, R, R**(K - i - 1)))
        tmp_perm = yv.permute(1, 0, 2)
        # Twiddle Factor multiplication
        D = torch.empty((R, R**i, R**(K - i - 1)), dtype=torch.complex128, device=x.device)
        tmp = torch.exp(-2.0j * torch.pi * ii_coord[:, :R**i] * jj_coord[:, :R**i] /
                     R**(i + 1))
        D[:] = torch.repeat_interleave(torch.reshape(tmp, (R, R**i, 1)), R**(K - i - 1), axis=2)
        tmp_twid = torch.reshape(tmp_perm, (N, )) * torch.reshape(D, (N, ))
        # Product with Butterfly
        y[:] = torch.reshape(dft_mat @ torch.reshape(tmp_twid, (R, R**(K - 1))),
                          (N, ))

stockham_fft_jit = torch.compile(stockham_fft)
