import prickle
import torch

@prickle.jit
def prickle_kernel(a, b, c, imgIn, imgOut, y1, y2, W, H):
    prickle.label('i')
    for i in range(W):
        y1[i, 0] = a[0] * imgIn[i, 0]
        y1[i, 1] = a[0] * imgIn[i, 1] + a[1] * imgIn[i, 0] + b[0] * y1[i, 0]
    for j in range(2, H):
        prickle.label('i')
        y1[:, j] = (a[0] * imgIn[:, j] + a[1] * imgIn[:, j-1] +
                    b[0] * y1[:, j-1] + b[1] * y1[:, j-2])

    prickle.label('i')
    for i in range(W):
        y2[i, -1] = 0.0
        y2[i, -2] = a[2] * imgIn[i, -1]
    for j in range(H-3, -1, -1):
        prickle.label('i')
        y2[:, j] = (a[2] * imgIn[:, j+1] + a[3] * imgIn[:, j+2] +
                    b[0] * y2[:, j+1] + b[1] * y2[:, j+2])

    prickle.label('i')
    prickle.label('j')
    imgOut[:, :] = c[0] * (y1[:, :] + y2[:, :])

    prickle.label('j')
    for j in range(H):
        y1[0, j] = a[4] * imgOut[0, j]
        y1[1, j] = a[4] * imgOut[1, j] + a[5] * imgOut[0, j] + b[0] * y1[0, j]
    for i in range(2, W):
        prickle.label('j')
        y1[i, :] = (a[4] * imgOut[i, :] + a[5] * imgOut[i-1, :] +
                    b[0] * y1[i-1, :] + b[1] * y1[i-2, :])

    prickle.label('j')
    for j in range(H):
        y2[W-1, j] = 0.0
        y2[W-2, j] = a[6] * imgOut[W-1, j]
    for i in range(W-3, -1, -1):
        prickle.label('j')
        y2[i, :] = (a[6] * imgOut[i+1, :] + a[7] * imgOut[i+2, :] +
                    b[0] * y2[i+1, :] + b[1] * y2[i+2, :])

    prickle.label('i')
    prickle.label('j')
    imgOut[:, :] = c[1] * (y1[:, :] + y2[:, :])

def kernel(alpha, imgIn):
    y1 = torch.empty_like(imgIn)
    y2 = torch.empty_like(imgIn)
    imgOut = torch.empty_like(imgIn)
    alpha = torch.tensor(alpha, dtype=imgIn.dtype)
    W, H = imgIn.shape
    k = (1.0 - prickle.exp(-alpha)) * (1.0 - prickle.exp(-alpha)) / (
        1.0 + alpha * prickle.exp(-alpha) - prickle.exp(2.0 * alpha))
    a = torch.empty(8, dtype=y1.dtype)
    b = torch.empty(2, dtype=y1.dtype)
    c = torch.empty_like(b)
    a[0] = a[4] = k
    a[1] = a[5] = k * prickle.exp(-alpha) * (alpha - 1.0)
    a[2] = a[6] = k * prickle.exp(-alpha) * (alpha + 1.0)
    a[3] = a[7] = -k * prickle.exp(-2.0 * alpha)
    b[0] = 2.0**(-alpha)
    b[1] = -prickle.exp(-2.0 * alpha)
    c[0] = c[1] = 1.0
    p = {
        'i': prickle.threads(W),
        'j': prickle.threads(H)
    }
    prickle_kernel(a, b, c, imgIn, imgOut, y1, y2, W, H, opts=prickle.par(p))
    return imgOut
