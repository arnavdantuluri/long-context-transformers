import torch

from einops import rearrange
from .bpt_pt import blockwise_compute_ffn
from .bpt_triton import forward as forward_triton

class FeedForwardWrapperNeoX(torch.nn.Module):
    def __init__(self, mlp, chunk_size, triton=True):
        super().__init__()
        self.cell = mlp
        self.triton = triton
        self.chunk_size = chunk_size
    
    def forward(self, hidden_states):
        '''hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states'''
        if not self.triton:
            hidden_states = blockwise_compute_ffn(self.cell.dense_h_to_4h, hidden_states, self.chunk_size)
            hidden_states = self.cell.act(hidden_states)
            hidden_states = blockwise_compute_ffn(self.cell.dense_4h_to_h, hidden_states, self.chunk_size)
        else:
            b = hidden_states.shape[0]
            hidden_states = rearrange(hidden_states, "b n d -> (b n) d")
            hidden_states = forward_triton(hidden_states, self.cell.dense_h_to_4h.weight.T.contiguous(), self.cell.dense_h_to_4h.bias)
            hidden_states = self.cell.act(hidden_states)
            hidden_states = forward_triton( hidden_states, self.cell.dense_4h_to_h.weight.T.contiguous(), self.cell.dense_4h_to_h.bias)
            hidden_states = rearrange(hidden_states, "(b n )d -> b n d", b=b)
        return hidden_states

class FeedForwardWrapperLLaMA(torch.nn.Module):
    def __init__(self, mlp, chunk_size, triton=True):
        super().__init__()
        self.cell = mlp
        self.triton = triton
        self.chunk_size = chunk_size
    
    def forward(self, hidden_states):
        '''return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))'''
        if not self.triton:
            hidden_states = blockwise_compute_ffn(self.cell.gate_proj, hidden_states, self.chunk_size) * blockwise_compute_ffn(self.cell.up_proj, hidden_states, self.chunk_size)
            hidden_states = self.cell.act_fn(hidden_states)
            hidden_states = blockwise_compute_ffn(self.cell.down_proj, hidden_states, self.chunk_size)
        else:
            b = hidden_states.shape[0]
            hidden_states = rearrange(hidden_states, "b n d -> (b n) d")
            hidden_states = forward_triton(hidden_states, self.cell.gate_proj.weight.T.contiguous(), self.cell.gate_proj.bias) * forward_triton(hidden_states, self.cell.up_proj.weight.T.contiguous(), self.cell.up_proj.bias)
            hidden_states = self.cell.act_fn(hidden_states)
            hidden_states = forward_triton( hidden_states, self.cell.down_proj.weight.T.contiguous(), self.cell.down_proj.bias)
            hidden_states = rearrange(hidden_states, "(b n )d -> b n d", b=b)
        return hidden_states

# Both MPT and Falcon already support memory saving methods for attention (Flash Attention) so there is no need for 
# BPT attention plugin we only need a plugin for the feed forward network
class FeedForwardWrapperMPT(torch.nn.Module):
    def __init__(self, mlp, chunk_size, triton=True):
        super().__init__()
        self.cell = mlp
        self.triton = triton
        self.chunk_size = chunk_size
    
    def forward(self, hidden_states):
        '''return self.down_proj(self.act(self.up_proj(x)))'''
        if not self.triton:
            hidden_states = blockwise_compute_ffn(self.cell.up_proj, hidden_states, self.chunk_size)
            hidden_states = self.cell.act(hidden_states)
            hidden_states = blockwise_compute_ffn(self.cell.down_proj, hidden_states, self.chunk_size)
        else:
            b = hidden_states.shape[0]
            hidden_states = rearrange(hidden_states, "b n d -> (b n) d")
            hidden_states = forward_triton(hidden_states, self.cell.up_proj.weight.T.contiguous(), self.cell.up_proj.bias)
            hidden_states = self.cell.act(hidden_states)
            hidden_states = forward_triton( hidden_states, self.cell.down_proj.weight.T.contiguous(), self.cell.down_proj.bias)
            hidden_states = rearrange(hidden_states, "(b n )d -> b n d", b=b)
        return hidden_states

# Falcon appears to follow the same nomenclature as Pythia and NeoX models
class FeedForwardWrapperFalcon(torch.nn.Module):
    def __init__(self, mlp, chunk_size, triton=True):
        super().__init__()
        self.cell = mlp
        self.triton = triton
        self.chunk_size = chunk_size
    
    def forward(self, hidden_states):
        '''x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x'''
        if not self.triton:
            hidden_states = blockwise_compute_ffn(self.cell.dense_h_to_4h, hidden_states, self.chunk_size)
            hidden_states = self.cell.act(hidden_states)
            hidden_states = blockwise_compute_ffn(self.cell.dense_4h_to_h, hidden_states, self.chunk_size)
        else:
            b = hidden_states.shape[0]
            hidden_states = rearrange(hidden_states, "b n d -> (b n) d")
            hidden_states = forward_triton(hidden_states, self.cell.dense_h_to_4h.weight.T.contiguous(), self.cell.dense_h_to_4h.bias)
            hidden_states = self.cell.act(hidden_states)
            hidden_states = forward_triton( hidden_states, self.cell.dense_4h_to_h.weight.T.contiguous(), self.cell.dense_4h_to_h.bias)
            hidden_states = rearrange(hidden_states, "(b n )d -> b n d", b=b)
        return hidden_states