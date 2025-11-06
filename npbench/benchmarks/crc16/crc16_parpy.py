import parpy
from parpy.builtin import convert
import torch

@parpy.jit
def crc16_kernel(data, poly, N, out):
    with parpy.gpu:
        crc = convert(0xFFFF, parpy.types.I32)
        for j in range(N):
            b = data[j]
            cur_byte = 0xFF & b
            for _ in range(8):
                if (crc & 0x0001) ^ (cur_byte & 0x0001):
                    crc = (crc >> 1) ^ poly
                else:
                    crc = crc >> 1
                cur_byte = cur_byte >> 1
        crc = (~crc & 0xFFFF)
        crc = (crc << 8) | ((crc >> 8) & 0xFF)
        out[0] = crc & 0xFFFF

def crc16(data, poly=0x8408):
    data = data.with_type(parpy.types.I32)
    N, = data.shape
    out = parpy.buffer.empty((1,), data.dtype, data.backend())
    crc16_kernel(data, poly, N, out, opts=parpy.par({}))
    return int(out.numpy()[0])
