import prickle

@prickle.jit
def crc16_kernel(data, poly, N, out):
    with prickle.gpu:
        crc = 0xFFFF
        for j in range(N):
            b = prickle.int32(data[j])
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
    N, = data.shape
    out = prickle.buffer.empty((1,), prickle.buffer.DataType("<i4"), data.backend)
    crc16_kernel(data, poly, N, out, opts=prickle.par({}))
    return int(out.numpy()[0])
