
import environment

import jax
import jax.numpy as jnp

from jax.sharding import PositionalSharding, Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils

import numpy as np

def run_attention():

    # define batch size, sequence length, and token embedding dimension
    BATCH_SIZE = 32
    SEQ_LEN = 128
    EMB_DIM = 64

    # define parameters of transformer block
    QK_DIM = 64
    V_DIM = 32
    NUM_HEADS = 4
    OUT_DIM = 32

    # define input to transformer encoder block
    X = jnp.ones((BATCH_SIZE, SEQ_LEN, EMB_DIM))

    # define the weights of single-headed attention
    W = jnp.ones((EMB_DIM, NUM_HEADS * (2 * QK_DIM + V_DIM)))

    # forward weights
    Wf = jnp.ones((NUM_HEADS * V_DIM, OUT_DIM))

    def attn(X, W, Wf):

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
        Y = jnp.dot(Y, Wf)
        
        return Y

    attn = jax.jit(attn)
    Y = attn(X, W, Wf)

    return


def main():
    run_attention()

if __name__ == "__main__":
    main()