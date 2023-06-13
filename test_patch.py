import torch
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import RotaryEmbedding
from .Mem_helpers.flash_attention_plugin import FlashAttentionWrapperWithRotary
from .Mem_helpers.bpt_attention_plugin import BPTAttentionWrapperWithRotary, FeedForwardWrapperNeoX

# model_path_or_name = "EleutherAI/pythia-1.4b"
model_path_or_name = "EleutherAI/pythia-160m-seed3"
tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
model = AutoModelForCausalLM.from_pretrained(model_path_or_name)

max_positions = 8192
tokenizer.pad_token = tokenizer.mask_token
model.config.max_position_embeddings=max_positions
tokenizer.model_max_length = max_positions
#tokenizer.model_max_length = max_positions

for each in model.gpt_neox.layers:
    original_emb = each.attention.rotary_emb
    #base = torch.sqrt(1/each.attention.rotary_emb.inv_freq[1]).cpu()
    each.attention.rotary_emb = RotaryEmbedding(each.attention.rotary_ndims,max_positions,10000)
    each.attention.bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            )
    each.attention = BPTAttentionWrapperWithRotary(each.attention, query_chunk_size=512, key_chunk_size=1024)
    each.mlp = FeedForwardWrapperNeoX(each.mlp, chunk_size=512)