from bpt_pytorch import blockwise_compute_attn as blockwise_compute_torch
from bpt_flax import blockwise_compute_attn as blockwise_compute_flax
from memeff import blockwise_compute_attn
from memrff_pt import memory_efficient_attention
import torch
import jax

if __name__ == "__main__":
    q = torch.rand(2, 1024, 16, 128)
    k = torch.rand(2, 1024, 16, 128)
    v = torch.rand(2, 1024, 16, 128)
    bias = torch.rand(2, 1, 1024, 1024)
    y_pt = blockwise_compute_torch(q, k, v, bias=bias, query_chunk_size=1024, key_chunk_size=1024)
    y_jax = blockwise_compute_flax(q.numpy(), k.numpy(), v.numpy(), bias=bias.numpy(), query_chunk_size=1024, key_chunk_size=1024)
    y_jax_mem = blockwise_compute_attn(q.numpy(), k.numpy(), v.numpy(), bias=bias.numpy(), query_chunk_size=1024, key_chunk_size=1024)
    y_pt_mem = memory_efficient_attention(q, k, v, q_bucket_size=1024, k_bucket_size=1024)
    diff = y_pt.numpy() - y_jax
    diff_mem_eff = y_jax_mem - y_pt_mem.numpy()
    assert (jax.lax.abs(diff) < 1).all()
    assert (jax.lax.abs(diff_mem_eff) < 1).all()