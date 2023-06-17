from bpt_pt import blockwise_compute_attn as blockwise_compute_torch
from bpt_flax import blockwise_compute_attn as blockwise_compute_flax
import torch
import jax

if __name__ == "__main__":
    q = torch.rand(2, 1024, 16, 128)
    k = torch.rand(2, 1024, 16, 128)
    v = torch.rand(2, 1024, 16, 128)
    bias = torch.rand(2, 1, 1024, 1024)
    y_pt = blockwise_compute_torch(q, k, v, bias=bias, query_chunk_size=1024, key_chunk_size=1024)
    y_jax = blockwise_compute_flax(q.numpy(), k.numpy(), v.numpy(), bias=bias.numpy(), query_chunk_size=1024, key_chunk_size=1024)
    diff = y_pt.numpy() - y_jax
    assert (jax.lax.abs(diff) < 1).all()
