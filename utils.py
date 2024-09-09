
import jax.numpy as jnp

def num_bytes(x):
    '''computes the number of bytes in a JAX numpy array x'''
    bytes_per_element = 0
    if x.dtype in [jnp.float32, jnp.int32, jnp.uint32]:
        bytes_per_element = 4
    else:
        raise NotImplementedError
    return x.size * bytes_per_element