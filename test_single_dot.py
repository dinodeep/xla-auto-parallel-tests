
import environment
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8 --xla_dump_to=/home/deepatel/dev/tests/dump_single_dot"

import jax
import jax.numpy as jnp

from jax.sharding import PositionalSharding, Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils

import numpy as np

def simple_sharding():

    B = jnp.ones((128, 16))
    C = jnp.ones((16, 32))

    device_mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ("batch", "model"))
    sharding_B = NamedSharding(device_mesh, P(None, None))
    sharding_C = NamedSharding(device_mesh, P(("batch", "model"), None))

    B = jax.device_put(B, sharding_B)
    C = jax.device_put(C, sharding_C)
    
    def mult(B, C):
        D = jnp.dot(B, C)
        return D

    mult = jax.jit(mult)
    E = mult(B, C)

    # jax.debug.visualize_array_sharding(B)
    # jax.debug.visualize_array_sharding(C)
    # jax.debug.visualize_array_sharding(E)


def main():
    simple_sharding()

if __name__ == "__main__":
    main()