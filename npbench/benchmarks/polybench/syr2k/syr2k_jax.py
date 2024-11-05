import jax
import jax.numpy as jnp


@jax.jit
def kernel(alpha, beta, C, A, B):
    for i in range(A.shape[0]):
        C = C.at[i, :i + 1].set(C[i, :i + 1] * beta)
        for k in range(A.shape[1]):
            C = C.at[i, :i + 1].set(C[i, :i + 1] + A[:i + 1, k] * alpha * B[i, k] +
                                    B[:i + 1, k] * alpha * A[i, k])
            
    return C
