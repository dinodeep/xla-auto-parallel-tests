
import environment
import utils

import jax
import jax.numpy as jnp

from jax.sharding import PositionalSharding, Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.pjit import AUTO

import numpy as np

def run_encoder():

    # construct the Mesh device
    mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ("batch", "model"))

    # define batch size, sequence length, and token embedding dimension
    BATCH_SIZE = 32
    SEQ_LEN = 128
    EMB_DIM = 64

    # define parameters of transformer block
    QK_DIM = 64
    V_DIM = 32
    FF_DIM = 64
    NUM_HEADS = 4

    # define input to transformer encoder block
    X = jax.core.ShapedArray((BATCH_SIZE, SEQ_LEN, EMB_DIM), np.float32)

    # define the weights of single-headed attention
    W = jax.core.ShapedArray((EMB_DIM, NUM_HEADS * (2 * QK_DIM + V_DIM)), np.float32)

    # linear weights for attention
    Wl = jax.core.ShapedArray((NUM_HEADS * V_DIM, EMB_DIM), np.float32)

    # feed forward weights for FFNN (result of FFNN is same shape)
    Wff1 = jax.core.ShapedArray((EMB_DIM, FF_DIM), np.float32)
    Wff2 = jax.core.ShapedArray((FF_DIM, EMB_DIM), np.float32)

    # get the sizes of all of the variables
    X_bytes = utils.num_bytes(X)
    W_bytes = utils.num_bytes(W)
    Wl_bytes = utils.num_bytes(Wl)
    Wff1_bytes = utils.num_bytes(Wff1)
    Wff2_bytes = utils.num_bytes(Wff2)

    # compute and display the total number of bytes in the parameters
    total_param_bytes = X_bytes + W_bytes + Wl_bytes + Wff1_bytes + Wff2_bytes
    print("Total Parameter Bytes: ", total_param_bytes)
    print("\tX:", X_bytes)
    print("\tW:", W_bytes)
    print("\tWl:", Wl_bytes)
    print("\tWff1:", Wff1_bytes)
    print("\tWff2:", Wff2_bytes)


    # output is same shape as input
    def encoder(X, W, Wl, Wff1, Wff2):

        # Sublayer 1: perform multi-head scaled dot product attention
        M = jnp.dot(X, W)

        # reshape to separate out heads
        M = jnp.reshape(M, (BATCH_SIZE, NUM_HEADS, SEQ_LEN, -1))

        # split into queries, keys, and values with heads separated
        # (B, H, S, (D_qk|D_v))
        Q, K, V = jnp.split(M, [QK_DIM, 2 * QK_DIM], axis=3)

        # compute scores using einsum
        S = jnp.einsum("bhsd,bhtd->bhst", Q, K)

        # perform elementwise operations to get attention weights
        A = S

        # now perform weighted combination of value vectors to do attention
        Y = jnp.einsum("bhst,bhtd->bhsd", A, V)

        # now concatenate the heads to get the final result
        Y = jnp.transpose(Y, [0, 2, 1, 3])
        Y = jnp.reshape(Y, (BATCH_SIZE, SEQ_LEN, -1))

        # to produce embeddings of dim OUT_DIM, push through linear layer
        Y = jnp.dot(Y, Wl)

        # Sublayer 2: perform FFNN
        # TODO: add biases as well here
        Y = jnp.dot(Y, Wff1)
        Y = jnp.dot(Y, Wff2)
        
        return Y

    with mesh:
        encoder= jax.jit(
            encoder,
            in_shardings=(AUTO(mesh),AUTO(mesh),AUTO(mesh),AUTO(mesh),AUTO(mesh),),
            out_shardings=AUTO(mesh)
        )
        lowered = encoder.lower(X, W, Wl, Wff1, Wff2).compile()

    return


def main():
    run_encoder()

if __name__ == "__main__":
    main()