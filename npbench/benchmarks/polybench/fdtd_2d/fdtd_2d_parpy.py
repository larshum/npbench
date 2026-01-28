import parpy

@parpy.jit
def kernel_wrap(ex, ey, hz, _fict_, TMAX):
    for t in range(TMAX):
        parpy.label('j')
        ey[0, :] = _fict_[t]

        parpy.label('i')
        parpy.label('j')
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])

        parpy.label('i')
        parpy.label('j')
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])

        parpy.label('i')
        parpy.label('j')
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])

def kernel(TMAX, ex, ey, hz, _fict_):
    NX, NY = ex.shape
    p = {
        'i': parpy.threads(NX-1),
        'j': parpy.threads(1024),
    }
    opts = parpy.par(p)
    opts.max_unroll_count = 0
    kernel_wrap(ex, ey, hz, _fict_, TMAX, opts=opts)

