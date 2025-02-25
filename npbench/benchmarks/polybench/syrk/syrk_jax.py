import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def kernel(alpha, beta, C, A):

    def loop_body(i, loop_vars):

        def inner_loop(k, loop_vars):
            alpha, C, A = loop_vars
            A_update_slice = jnp.where(jnp.arange(A.shape[0]) < i + 1, A[:, k], 0.0)
            A_update_slice *= alpha * A[i, k]

            C_update_slice = jnp.where(jnp.arange(C.shape[1]) < i + 1, C[i, :], 0.0)
            C_update_slice += A_update_slice
            C_update_slice = jnp.where(jnp.arange(C.shape[1]) < i + 1, C_update_slice, C[i, :])

            C = lax.dynamic_update_slice(C, C_update_slice[None, :], (i, 0))
            return alpha, C, A

        alpha, beta, C, A = loop_vars

        C_slice = jnp.where(jnp.arange(C.shape[1]) < i + 1, C[i, :], 0.0)
        C_slice = C_slice * beta
        C_slice = jnp.where(jnp.arange(C.shape[1]) < i + 1, C_slice, C[i, :])
        C = lax.dynamic_update_slice(C, C_slice[None, :], (i, 0))
        
        _, C, _ = lax.fori_loop(0, A.shape[1], inner_loop, (alpha, C, A))

        return alpha, beta, C, A
    
    _, _, C, _ = lax.fori_loop(0, A.shape[0], loop_body, (alpha, beta, C, A))

    return C
