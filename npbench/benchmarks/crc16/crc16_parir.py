import parir
import torch


@parir.jit
def crc16_kernel(data, poly, N, out):
    # Wrap in "parallel" loop to write sequential loop in Parir
    parir.label('i')
    for i in range(1):
        crc = parir.int32(0xFFFF)
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
    data = data.to(dtype=torch.int32)
    N, = data.shape
    out = torch.empty(1, dtype=torch.int32, device='cuda')
    p = {'i': [parir.threads(2)]}
    crc16_kernel(data, poly, N, out, parallelize=p)
    return int(out[0])
