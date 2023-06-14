import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn 
from transformers.activations import ACT2FN

class GPTNeoXMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(512, 2048)
        self.dense_4h_to_h = nn.Linear(2048, 512)
        self.act = ACT2FN["gelu"]

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states

import torch
from torch.utils.checkpoint import checkpoint
from .utils import dynamic_slice, map_pt, scan
import math

def _query_chunk_attention(query_idx, query, key, value,
                           mask, bias, key_chunk_size=4096,
                           mask_calc_fn=None,
                           bias_calc_fn=None,
                           weights_calc_fn=None,
                           calc_fn_data=None):
    num_kv, num_heads, k_features = key.shape[-3:]
    v_features = value.shape[-1]
    num_q = query.shape[-3]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / math.sqrt(k_features)

    def summarize_chunk(key_idx, query, key, value, mask, bias):
        attn_weights = torch.einsum('...qhd,...khd->...qhk', query, key)
        if bias_calc_fn is not None:
            bias = bias_calc_fn(query_idx, key_idx, bias, attn_weights, calc_fn_data)
        if bias is not None:
            bias = torch.einsum('...hqk->...qhk', bias)
            attn_weights = attn_weights + bias
        if mask_calc_fn is not None:
            mask = mask_calc_fn(query_idx, key_idx, mask, attn_weights, calc_fn_data)
        if mask is not None:
            big_neg = torch.finfo(attn_weights.dtype).min
            big_neg = torch.tensor(big_neg, device=mask.device, dtype=torch.float32)
            mask = torch.einsum('...hqk->...qhk', mask)
            attn_weights = torch.where(mask, attn_weights, big_neg)
        if weights_calc_fn is not None:
            attn_weights = weights_calc_fn(query_idx, key_idx, attn_weights, calc_fn_data)
        max_score, _ = torch.max(attn_weights, -1, keepdim=True)
        max_score = max_score.detach()
        exp_weights = torch.exp(attn_weights - max_score)
        exp_values = torch.einsum('...vhf,...qhv->...qhf', value, exp_weights)
        max_score = torch.einsum('...qhk->...qh', max_score)
        return exp_values, exp_weights.sum(dim=-1), max_score

    def chunk_scanner(chunk_idx):
        key_chunk = dynamic_slice(key, tuple([0] * (key.ndim - 3)) + (chunk_idx, 0, 0),
                                  tuple(key.shape[:-3]) + (key_chunk_size, num_heads, k_features))
        value_chunk = dynamic_slice(value, tuple([0] * (value.ndim - 3)) + (chunk_idx, 0, 0),
                                    tuple(value.shape[:-3]) + (key_chunk_size, num_heads, v_features))

        if bias is None:
            bias_chunk = None
        elif bias.shape[-1] == 1:
            bias_chunk = bias
        elif bias.shape[-1] == num_kv:
            bias_chunk = dynamic_slice(bias, tuple([0] * (bias.ndim - 3)) + (0, 0, chunk_idx),
                                       tuple(bias.shape[:-3]) + (bias.shape[-3], bias.shape[-2], key_chunk_size))
        else:
            raise TypeError(f'bias.shape[-1] == {bias.shape[-1]} must broadcast with key.shape[-3] == {num_kv}')

        if mask is None:
            mask_chunk = None
        elif mask.shape[-1] == 1:
            mask_chunk = mask
        elif mask.shape[-1] == num_kv:
            mask_chunk = dynamic_slice(mask, tuple([0] * (mask.ndim - 3)) + (0, 0, chunk_idx),
                                       tuple(mask.shape[:-3]) + (mask.shape[-3], mask.shape[-2], key_chunk_size))
        else:
            raise TypeError(f'bias.shape[-1] == {bias.shape[-1]} must broadcast with key.shape[-3] == {num_kv}')

        return checkpoint(summarize_chunk, chunk_idx, query, key_chunk, value_chunk, mask_chunk, bias_chunk)

    chunk_values, chunk_weights, chunk_max = map_pt(
        chunk_scanner, xs=torch.arange(0, num_kv, key_chunk_size))

    global_max, _ = torch.max(chunk_max, 0, keepdim=True)
    max_diffs = torch.exp(chunk_max - global_max)
    chunk_values *= torch.unsqueeze(max_diffs, -1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(dim=0)
    all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)
    return all_values / all_weights


def blockwise_compute_attn(query, key, value,
                                    mask=None, bias=None,
                                    query_chunk_size=1024,
                                    key_chunk_size=4096,
                                    bias_calc_fn=None,
                                    mask_calc_fn=None,
                                    weights_calc_fn=None,
                                    calc_fn_data=None):
    """Computes efficient dot-product attention given query, key, and value.
      This is efficient version of attention presented in
      https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
      Note: query, key, value needn't have any batch dimensions.
      Args:
        query: queries for calculating attention with shape of
          `[batch..., q_length, num_heads, qk_depth_per_head]`.
        key: keys for calculating attention with shape of
          `[batch..., kv_length, num_heads, qk_depth_per_head]`.
        value: values to be used in attention with shape of
          `[batch..., kv_length, num_heads, v_depth_per_head]`.
        bias: bias for the attention weights. This should be broadcastable to the
          shape `[batch..., num_heads, q_length, kv_length]`.
          This can be used for incorporating padding masks, proximity bias, etc.
        mask: mask for the attention weights. This should be broadcastable to the
          shape `[batch..., num_heads, q_length, kv_length]`.
          Attention weights are masked out if their corresponding mask value
          is `False`.
        query_chunk_size: int: query chunks size
        key_chunk_size: int: key chunks size
        bias_calc_fn: a bias calculation callback for each chunk, of form
          `(q_offset, k_offset, bias_chunk, attn_weights, calc_fn_data) -> bias`.
          This can be used for incorporating causal masks, padding masks,
          proximity bias, etc.
        mask_calc_fn: a mask calculation callback for each chunk, of form
          `(q_offset, k_offset, mask_chunk, attn_weights, calc_fn_data) -> mask`.
          This can be used for incorporating causal or other large masks.
          Attention weights are masked out if their corresponding mask value
          is `False`.
        weights_calc_fn: a general attn_weights callback for each chunk, of form
          `(q_offset, k_offset, attn_weights, calc_fn_data) -> attn_weights`.
          attn_weights has shape of
          `[batch..., q_chunk_size, num_heads, k_chunk_size]`.
          This can be used to implement complex weights processing in a memory
          efficient way.
        calc_fn_data: optional pure data to pass to each per-chunk call of
          bias_calc_fn, mask_calc_fn, and weights_calc_fn.
        weights_calc_data: pure_data to pass with each call to weights_calc_fn
      Returns:
        Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
      """
    num_q, num_heads, q_features = query.shape[-3:]
    num_kv = key.shape[-3]

    def chunk_scanner(chunk_idx, _):
        query_chunk = dynamic_slice(query, tuple([0] * (query.ndim - 3)) + (chunk_idx, 0, 0),
                                    tuple(query.shape[:-3]) + (min(query_chunk_size, num_q), num_heads, q_features))

        if mask is None:
            mask_chunk = None
        elif mask.shape[-2] == 1:
            mask_chunk = mask
        elif mask.shape[-2] == num_q:
            mask_chunk = dynamic_slice(mask, tuple([0] * (mask.ndim - 3)) + (0, chunk_idx, 0),
                                       tuple(mask.shape[:-3]) + (mask.shape[-3], min(query_chunk_size, num_q), mask.shape[-1]))
        else:
            raise TypeError(f'mask.shape[-2] == {mask.shape[-2]} must broadcast with query.shape[-3] == {num_q}')

        if bias is None:
            bias_chunk = None
        elif bias.shape[-2] == 1:
            bias_chunk = bias
        elif bias.shape[-2] == num_q:
            bias_chunk = dynamic_slice(bias, tuple([0] * (bias.ndim - 3)) + (0, chunk_idx, 0),
                                       tuple(bias.shape[:-3]) + (bias.shape[-3], min(query_chunk_size, num_q), bias.shape[-1]))
        else:
            raise TypeError(f'bias.shape[-2] == {bias.shape[-2]} must broadcast with query.shape[-3] == {num_q}')
        return (chunk_idx + query_chunk_size,
                _query_chunk_attention(chunk_idx, query_chunk, key, value, mask_chunk, bias_chunk, key_chunk_size=key_chunk_size,
                                       bias_calc_fn=bias_calc_fn, mask_calc_fn=mask_calc_fn,
                                       weights_calc_fn=weights_calc_fn, calc_fn_data=calc_fn_data))

    _, res = scan(chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size))
    rl = [res[i] for i in range(res.shape[0])]
    return torch.cat(rl, dim=-3)

def blockwise_compute_ffn(cell, inputs, chunk_size):
    inputs = torch.split(inputs, chunk_size, dim=-2)
    num_q = len(inputs)

    def ffn(cell, _, hidden_states):
        outputs = cell(hidden_states)
        return outputs
    
    outputs = []
    for i in range(num_q):
        outputs.append(ffn(cell, None, inputs[i]))
    
    res = torch.concat(outputs, dim=-2)
    # res = rearrange(res, 'n b c d -> b (n c) d')
    return res

import torch
from functools import partial
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# regular attention

def attention(
    q, k, v,
    mask = None,
    causal = False,
    attn_bias = None,
    **kwargs
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    sim = einsum('b h i d, b h j d -> b h i j', q, k)

    if exists(attn_bias):
        sim = sim + attn_bias

    mask_value = -torch.finfo(sim.dtype).max

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        sim = sim.masked_fill(~mask, mask_value)

    if causal:
        i, j = sim.shape[-2:]
        mask = torch.ones(i, j, device = q.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(mask, mask_value)

    sim = sim - sim.amax(dim = -1, keepdim = True).detach()
    attn = sim.softmax(dim = -1)

    out = einsum('b h i j, b h j d -> b h i d', attn, v)
    return out

# memory efficient attention

def summarize_qkv_chunk(q, k, v, mask, attn_bias_chunk, causal, qk_start_indices, dropout):
    q_start_index, k_start_index, q_chunk_size, k_chunk_size, device = *qk_start_indices, q.shape[-2], k.shape[-2], q.device

    weight = einsum('b h i d, b h j d -> b h i j', q, k)

    if exists(attn_bias_chunk):
        weight = weight + attn_bias_chunk

    mask_value = -torch.finfo(weight.dtype).max

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        weight = weight.masked_fill(~mask, mask_value)

    if causal and q_start_index < (k_start_index + k_chunk_size - 1):
        causal_mask = torch.ones((q_chunk_size, k_chunk_size), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
        weight = weight.masked_fill(causal_mask, mask_value)

    weight_max = weight.amax(dim = -1, keepdim = True).detach()
    weight = weight - weight_max

    exp_weight = weight.exp()

    exp_weight = F.dropout(exp_weight, p = dropout)

    weighted_value = einsum('b h i j, b h j d -> b h i d', exp_weight, v)

    return exp_weight.sum(dim = -1), weighted_value, rearrange(weight_max, '... 1 -> ...')

checkpointed_summarize_qkv_chunk = partial(checkpoint, summarize_qkv_chunk)

def memory_efficient_attention(
    q, k, v,
    mask = None,
    causal = False,
    attn_bias = None,
    q_bucket_size = 512,
    k_bucket_size = 1024,
    eps = 1e-8,
    dropout = 0.,
    training = False
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # function

    needs_backwards = q.requires_grad or k.requires_grad or v.requires_grad
    summarize_qkv_fn = checkpointed_summarize_qkv_chunk if needs_backwards else summarize_qkv_chunk

    # chunk all the inputs

    q_chunks = q.split(q_bucket_size, dim = -2)
    k_chunks = k.split(k_bucket_size, dim = -2)
    v_chunks = v.split(k_bucket_size, dim = -2)
    mask_chunks = mask.split(k_bucket_size, dim = -1) if exists(mask) else ((None,) * len(k_chunks))

    if exists(attn_bias):
        i, j = attn_bias.shape[-2:]
        attn_bias_chunks = attn_bias.split(q_bucket_size, dim = -2)
        attn_bias_chunks = list(map(lambda t: t.split(k_bucket_size, dim = -1), attn_bias_chunks))

    # loop through all chunks and accumulate

    out = []
    for q_index, q_chunk in enumerate(q_chunks):
        exp_weights = []
        weighted_values = []
        weight_maxes = []

        for k_index, (k_chunk, v_chunk, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):
            q_start_index = q_index * q_bucket_size
            k_start_index = k_index * k_bucket_size

            if causal and k_start_index > (q_start_index + q_chunk.shape[-2] - 1):
                # if chunk is to be all masked out causally, skip
                continue

            attn_bias_chunk = attn_bias_chunks[q_index][k_index] if exists(attn_bias) else None

            exp_weight_chunk, weighted_value_chunk, weight_max_chunk = summarize_qkv_fn(
                q_chunk,
                k_chunk,
                v_chunk,
                mask_chunk,
                attn_bias_chunk,
                causal,
                (q_start_index, k_start_index),
                dropout if training else 0.
            )

            exp_weights.append(exp_weight_chunk)
            weighted_values.append(weighted_value_chunk)
            weight_maxes.append(weight_max_chunk)

        weight_maxes = torch.stack(weight_maxes, dim = -1)

        weighted_values = torch.stack(weighted_values, dim = -1)
        exp_weights = torch.stack(exp_weights, dim = -1)

        global_max = weight_maxes.amax(dim = -1, keepdim = True)
        renorm_factor = (weight_maxes - global_max).exp().detach()

        exp_weights = exp_weights * renorm_factor
        weighted_values = weighted_values * rearrange(renorm_factor, '... c -> ... 1 c')

        all_values = weighted_values.sum(dim = -1)
        all_weights = exp_weights.sum(dim = -1)

        normalized_values = all_values / (rearrange(all_weights, '... -> ... 1') + eps)
        out.append(normalized_values)

    return torch.cat(out, dim = -2)

if __name__ == "__main__":
    # Blocked mem stuff
    q = torch.rand(2, 512, 16, 128)
    k = torch.rand(2, 2048, 16, 128)
    v = torch.rand(2, 2048, 16, 128)
    bias = torch.rand(2, 1, 512, 2048)

    # Blocked FFN Stuff
    x = torch.rand(2, 256, 512)
    cell = GPTNeoXMLP()
    y_pt_mem = blockwise_compute_attn(q, k, v, bias=bias, query_chunk_size=512, key_chunk_size=512)
    y_pt_ffn = blockwise_compute_ffn(cell, x, 256)
    print(y_pt_ffn.shape)