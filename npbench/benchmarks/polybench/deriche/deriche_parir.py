import parir
import torch

@parir.jit
def parir_kernel(a, b, c, imgIn, imgOut, y1, y2, W, H):
    for i in range(W):
        y1[i, 0] = a[0] * imgIn[i, 0]
        y1[i, 1] = a[0] * imgIn[i, 1] + a[1] * imgIn[i, 0] + b[0] * y1[i, 0]
    for jx in range(2, H):
        for i in range(W):
            y1[i, jx] = (a[0] * imgIn[i, jx] + a[1] * imgIn[i, jx-1] +
                         b[0] * y1[i, jx-1] + b[1] * y1[i, jx-2])

    for i in range(W):
        y2[i, H-1] = 0.0
        y2[i, H-2] = a[2] * imgIn[i, H-1]
    for jx in range(0, H-2):
        for i in range(W):
            j = (H-3) - jx
            y2[i, j] = (a[2] * imgIn[i, j+1] + a[3] * imgIn[i, j+2] +
                        b[0] * y2[i, j+1] + b[1] * y2[i, j+2])

    for i in range(W):
        for j in range(H):
            imgOut[i, j] = c[0] * (y1[i, j] + y2[i, j])

    for j in range(H):
        y1[0, j] = a[4] * imgOut[0, j]
        y1[1, j] = a[4] * imgOut[1, j] + a[5] * imgOut[0, j] + b[0] * y1[0, j]
    for ix in range(2, W):
        for j in range(H):
            y1[ix, j] = (a[4] * imgOut[ix, j] + a[5] * imgOut[ix-1, j] +
                         b[0] * y1[ix-1, j] + b[1] * y1[ix-2, j])

    for j in range(H):
        y2[W-1, j] = 0.0
        y2[W-2, j] = a[6] * imgOut[W-1, j]
    for ix in range(0, W-2):
        for j in range(H):
            i = (W-3) - ix
            y2[i, j] = (a[6] * imgOut[i+1, j] + a[7] * imgOut[i+2, j] +
                        b[0] * y2[i+1, j] + b[1] * y2[i+2, j])

    for i in range(W):
        for j in range(H):
            imgOut[i, j] = c[1] * (y1[i, j] + y2[i, j])

def kernel(alpha, imgIn):
    y1 = torch.empty_like(imgIn)
    y2 = torch.empty_like(imgIn)
    imgOut = torch.empty_like(imgIn)
    W, H = imgIn.shape
    k = (1.0 - parir.exp(-alpha)) * (1.0 - parir.exp(-alpha)) / (
        1.0 + alpha * parir.exp(-alpha) - parir.exp(2.0 * alpha))
    a = torch.empty(8, dtype=y1.dtype, device=y1.device)
    b = torch.empty(2, dtype=y1.dtype, device=y1.device)
    c = torch.empty_like(b)
    a[0] = a[4] = k
    a[1] = a[5] = k * parir.exp(-alpha) * (alpha - 1.0)
    a[2] = a[6] = k * parir.exp(-alpha) * (alpha + 1.0)
    a[3] = a[7] = -k * parir.exp(-2.0 * alpha)
    b[0] = 2.0**(-alpha)
    b[1] = -parir.exp(-2.0 * alpha)
    c[0] = c[1] = 1.0
    p = {
        'i': [parir.threads(W)],
        'j': [parir.threads(H)]
    }
    parir_kernel(a, b, c, imgIn, imgOut, y1, y2, W, H, parallelize=p)
    return imgOut
