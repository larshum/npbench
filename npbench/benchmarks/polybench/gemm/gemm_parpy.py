import parpy
from parpy.types import F64, I64
import pathlib


M = parpy.types.symbol()
N = parpy.types.symbol()
K = parpy.types.symbol()

@parpy.external("cublas_gemm_f64", parpy.CompileBackend.Cuda, parpy.Target.Host, header="<cublas_helper.h>")
def cublas_gemm_f64(
        M: I64,
        N: I64,
        K: I64,
        alpha: F64,
        A: parpy.types.buffer(F64, [M, K]),
        B: parpy.types.buffer(F64, [K, N]),
        beta: F64,
        C: parpy.types.buffer(F64, [M, N])):
    pass

@parpy.jit
def inplace_gemm_wrap(
        alpha: F64,
        beta: F64,
        C: parpy.types.buffer(F64, [M, N]),
        A: parpy.types.buffer(F64, [M, K]),
        B: parpy.types.buffer(F64, [K, N])
):
    with parpy.gpu:
        cublas_gemm_f64(M, N, K, alpha, A, B, beta, C)

def kernel(alpha, beta, C, A, B):
    if C.backend() == parpy.CompileBackend.Cuda:
        # Add the current directory to the include path, so our file "cublas.h"
        # is found, and add a flag to link with the cuBLAS library.
        opts = parpy.par({})
        opts.includes += [str(pathlib.Path(__file__).parent.resolve())]
        opts.extra_flags += ["-lcublas"]
        inplace_gemm_wrap(alpha, beta, C, A, B, opts=opts)
    else:
        raise RuntimeError(f"GEMM is only implemented for CUDA")
