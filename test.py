
import environment

import jax
import jax.numpy as jnp

from jax.sharding import PositionalSharding, Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.pjit import AUTO

import numpy as np

def num_bytes(x):
    bytes_per_element = 0
    if x.dtype in [jnp.float32, jnp.int32, jnp.uint32]:
        bytes_per_element = 4
    else:
        raise NotImplementedError
    return x.size * bytes_per_element

def simple_sharding():

    A = jax.core.ShapedArray((24, 48, 128), np.float32)
    B = jax.core.ShapedArray((128, 16), np.float32)
    C = jax.core.ShapedArray((16, 32), np.float32)

    device_mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ("batch", "model"))
    sharding_A = NamedSharding(device_mesh, P(("batch", "model"), None, None))
    sharding_B = NamedSharding(device_mesh, P(None, ("batch", "model")))

    # A = jax.device_put(A, sharding_A)
    # B = jax.device_put(B, sharding_B)
    # jax.debug.visualize_array_sharding(A)
    # jax.debug.visualize_array_sharding(B)

    def mult(A, B, C):
        D = jnp.dot(A, B)
        E = jnp.dot(D, C)


        return E

    with device_mesh:
        mult = jax.jit(
            mult,
            in_shardings=(AUTO(device_mesh), AUTO(device_mesh), AUTO(device_mesh)),
            out_shardings=AUTO(device_mesh)
        )
        lowered = mult.lower(A, B, C).compile()
    # E = mult(A, B, C)
    # jax.debug.visualize_array_sharding(C)


def main():
    simple_sharding()

if __name__ == "__main__":
    main()