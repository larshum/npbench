import parpy

@parpy.jit
def kernel_helper(path, N):
    for k in range(N):
        parpy.label('i')
        for i in range(N):
            parpy.label('j')
            path[i,:] = parpy.builtin.minimum(path[i,:], path[i,k] + path[k,:])

def kernel(path):
    N, N = path.shape
    p = {
        'i': parpy.threads(N),
        'j': parpy.threads(N)
    }
    kernel_helper(path, N, opts=parpy.par(p))
