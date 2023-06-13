import functools
import json
import math
from functools import partial
from typing import Callable, NamedTuple, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange, einsum
from flax.linen import combine_masks, make_causal_mask
from jax import lax
import torch
import opt_einsum
from jax import numpy as jnp

MASK_VALUE = -1e10

def scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, np.stack(ys)

def get_gradient_checkpoint_policy(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        # 'dots_saveable': jax.checkpoint_policies.dots_saveable,
        # 'dots_with_no_batch_dims_saveable': jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    }[name]

def _chunk_attention_bias(query_chunk_size, key_chunk_size,
            bias, deterministic, attn_dropout, attn_pdrop, causal_mask,
            query_chunk_idx, key_chunk_idx):
    query_offset = query_chunk_idx * query_chunk_size
    key_offset = key_chunk_idx * key_chunk_size
    chunk_bias = jnp.zeros((1, 1, 1, 1))
    if bias is not None:
        chunk_bias = lax.dynamic_slice(
            bias,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(*bias.shape[:2], min(bias.shape[-2], query_chunk_size), min(bias.shape[-1], key_chunk_size)),
        )

    if causal_mask:
        query_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(query_chunk_size, 1), dimension=0)
        key_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, key_chunk_size), dimension=1)
        offset = query_offset - key_offset
        query_idx += offset
        causal_mask_value = (query_idx < key_idx) * MASK_VALUE
        chunk_bias += causal_mask_value.reshape(1, 1, *causal_mask_value.shape)

    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_slice = lax.dynamic_slice(
            attn_dropout,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *attn_dropout.shape[:2],
                min(attn_dropout.shape[-2], query_chunk_size),
                min(attn_dropout.shape[-1], key_chunk_size),
            ),
        )
        chunk_bias -= attn_dropout_slice * 1e6
    return chunk_bias

class Carry(NamedTuple):
    numerator: jax.Array
    denominator: jax.Array
    max_so_far: jax.Array

def blockwise_compute_attn(query, key, value,
        bias=None,
        deterministic=False,
        dropout_rng=None,
        attn_pdrop=0.0,
        causal_mask=True,
        query_chunk_size=None,
        key_chunk_size=None,
        dtype=jnp.float32,
        policy='nothing_saveable',
        precision=lax.Precision.HIGHEST,
        prevent_cse=False,):
    q_len = query.shape[1]
    kv_len = key.shape[1]
    query = rearrange(query, 'b (n c) h q -> b n c h q', c=query_chunk_size)
    key, value = map(lambda t: rearrange(t, 'b (n c) h v -> b n c h v', c=key_chunk_size), (key, value))
    query, key, value = map(lambda t: rearrange(t, 'b n c h d -> n b c h d'), (query, key, value))
    num_q, batch, _, num_heads, dim_per_head = query.shape
    num_kv, _, _, _, _ = key.shape
    print("num kv", num_kv)

    for bias_dim, broadcast_dim in zip(bias.shape, (batch, num_heads, q_len, kv_len)):
        assert bias_dim == 1 or bias_dim == broadcast_dim
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None

    _chunk_bias_fn = functools.partial(
        _chunk_attention_bias,
        query_chunk_size, key_chunk_size,
        bias, deterministic, attn_dropout, attn_pdrop, causal_mask)

    def _query_chunk_attention(args):
        query_chunk, query_chunk_idx = args
        @functools.partial(jax.checkpoint, prevent_cse=prevent_cse,
                           policy=get_gradient_checkpoint_policy(policy))
        def summarize_chunk(carry, args):
            key_chunk, value_chunk, key_chunk_idx = args
            print(key_chunk_idx)
            (numerator, denominator, prev_max_score) = carry
            attn_weights = jnp.einsum('bqhd,bkhd->bqhk', query_chunk, key_chunk )
            bias_chunk = _chunk_bias_fn(query_chunk_idx, key_chunk_idx)
            bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)
            attn_weights = attn_weights + bias_chunk

            max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
            max_score = jnp.maximum(prev_max_score, max_score)
            max_score = jax.lax.stop_gradient(max_score)
            exp_weights = jnp.exp(attn_weights - max_score)
            exp_values = opt_einsum.contract(
                'bqhv,bvhf->bqhf', exp_weights, value_chunk
            )
            correction = jnp.exp(prev_max_score - max_score)
            numerator = numerator * correction + exp_values
            denominator = denominator * correction + exp_weights.sum(axis=-1, keepdims=True)
            return Carry(numerator, denominator, max_score), None

        init_carry = Carry(
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=dtype),
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=dtype),
            (-jnp.inf) * jnp.ones((batch, query_chunk_size, num_heads, 1), dtype=dtype),
        )
        with jax.disable_jit():
            (numerator, denominator, max_score), _ = lax.scan(
                summarize_chunk, init_carry, xs=(key, value, jnp.arange(0, num_kv))
            )
        outputs = (numerator / denominator).astype(dtype)
        return outputs
    with jax.disable_jit():
        _, res = lax.scan(
            lambda _,x: ((), _query_chunk_attention(x)),
            (), xs=(query, jnp.arange(0, num_q))
        )
    res = rearrange(res, 'n b c h d -> b (n c) h d')
    return res

def blockwise_compute_ffn(cell, inputs, chunk_size, deterministic, policy, prevent_cse):
    inputs = rearrange(inputs, 'b (n c) d -> b n c d', c=chunk_size)
    inputs = rearrange(inputs, 'b n c d -> n b c d')
    num_q, _, _, _ = inputs.shape
    def ffn(cell, _, hidden_states):
        outputs = cell.forward_ffn(hidden_states, deterministic=deterministic)
        return _, outputs
    ffn_remat = nn.remat(
        ffn,
        variables="params",
        rngs={"params" : False},
        prevent_cse=prevent_cse,
        policy=get_gradient_checkpoint_policy(policy),
    )
    _, res = nn.scan(
        ffn_remat,
        variable_broadcast="params",
        split_rngs={"params": False},
        in_axes=0,
        out_axes=0,
        length=num_q,
    )(cell, None, inputs)
    res = rearrange(res, 'n b c d -> b (n c) d')
    return res

if __name__ == "__main__":
    q = torch.rand(2, 1024, 16, 128).numpy()
    k = torch.rand(2, 2048, 16, 128).numpy()
    v = torch.rand(2, 2048, 16, 128).numpy()
    bias = torch.rand(2, 1, 1024, 2048).numpy()
    with jax.disable_jit():
        y = blockwise_compute_attn(q, k, v, bias=bias, query_chunk_size=256, key_chunk_size=256)
    print(y)