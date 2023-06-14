#@title Longformer Plug-in
import torch
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from flash_attn.modules.mha import FlashSelfAttention
import torch.nn as nn

def set_global_attention_indices(model, global_attention_indices):
    for each in model.gpt_neox.layers:
        each.attention.set_global_attention_indices(global_attention_indices)


class LongformerAttentionWrapperWithRotary(torch.nn.Module):
    def __init__(self, attention, max_seqlen = 8192, config = None, layer_id = 0):
        super().__init__()
        self.max_seqlen = max_seqlen
        self.attention = attention
        self.dropout_p = 0.0
        self.is_global_attn = config.is_global_attn
        self.dtype = config.dtype
        self.layer_id = layer_id
        self.is_global_attn = config.is_global_attn
        self.max_global_tokens = config.max_global_tokens
        self.config = config
        if config.attn_type == "flash_attn":
            self.flash_self_attention = FlashSelfAttention(causal = True, softmax_scale = 1/self.attention.norm_factor)
    
    def set_global_attention_indices(self, global_attention_indices):
        # make sure you provide input in the format of a 2D list
        self.global_attention_indices = global_attention_indices

    def get_global_tokens(self, key, value, i):
        # getting the global tokens you want to attend to before ith token.
        batch_size, num_heads, _, head_dim = key.shape
        global_attention_mask = []
        global_key = []
        global_value = []
        for j in range(batch_size):
            causal_index = torch.tensor([each for each in self.global_attention_indices[j] if each < i]).long().to(key.device)
            causal_index = causal_index[:self.max_global_tokens]
            tmp_key = torch.index_select(key[j], 1, causal_index).unsqueeze(0)
            tmp_value = torch.index_select(value[j], 1, causal_index).unsqueeze(0)
            tmp_mask = torch.zeros(1, 1, 1, self.max_global_tokens).type(self.dtype).to(key.device)
            if self.max_global_tokens > len(causal_index):
                pad = torch.zeros(1, num_heads, self.max_global_tokens - len(causal_index), head_dim).type(self.dtype).to(key.device)
                tmp_key = torch.cat([tmp_key, pad], dim = 2)
                tmp_value = torch.cat([tmp_value, pad], dim = 2)
                tmp_mask[:,:,:,len(causal_index):] = torch.finfo(self.dtype).min
            global_key.append(tmp_key)
            global_value.append(tmp_value)
            global_attention_mask.append(tmp_mask)
        global_key = torch.cat(global_key)
        global_value = torch.cat(global_value)
        global_attention_mask = torch.cat(global_attention_mask)
        return global_key, global_value, global_attention_mask
        
    def long_attn(self, query, key, value, attention_mask, head_mask):
        batch_size, num_heads, seq_len, head_dim = key.shape
        window_size = self.config.attention_window[self.layer_id]
        if seq_len <= window_size:
            return self._attn(query, key.mean(1).unsqueeze(1), value.mean(1).unsqueeze(1), attention_mask, head_mask)
        else:
            if query.size() == key.size():
                if self.config.attn_type == "flash_attn":
                    # this is already a global attention for context so it doesn't matter if it's global attention or not
                    qkv = torch.concat([query.unsqueeze(2), key.unsqueeze(2), value.unsqueeze(2)], dim = 2).permute(0, 3, 2, 1, 4).half()
                    attn_output = self.flash_self_attention(qkv)
                    attn_output = attn_output.transpose(1, 2)
                    return attn_output, None
                #Modify to use the kernels built for faster sliding attention
                elif self.config.attn_type == "step_attn":
                    partial_attn_output, _ = self._attn(query[:,:,:window_size, :], key[:,:,:window_size, :].mean(1).unsqueeze(1), value[:,:, : window_size, :].mean(1).unsqueeze(1), attention_mask[:,:,:,:window_size], head_mask)
                    attn_output = [partial_attn_output]
                    for i in range(seq_len - window_size):
                        combined_key = key[:,:,i+1:window_size+i+1, :]
                        combined_value = value[:,:, i+1:window_size+i+1, :]
                        combined_attention_mask = attention_mask[:,:,:,i+1:window_size+i+1]
                        if self.is_global_attn:
                            global_key, global_value, global_attention_mask = self.get_global_tokens(key, value, i)
                            combined_key = torch.cat([global_key, combined_key], dim = 2)
                            combined_value = torch.cat([global_value, combined_value], dim = 2)
                            combined_attention_mask = torch.cat([global_attention_mask, combined_attention_mask], dim = -1)
                        print("Combined key", combined_key.shape)
                        partial_attn_output, _ = self._attn(query[:,:,i+window_size:window_size+i+1, :], combined_key.mean(1).unsqueeze(1), combined_value.mean(1).unsqueeze(1), combined_attention_mask, head_mask)
                        attn_output.append(partial_attn_output)
                    attn_output = torch.cat(attn_output, dim = 2)
                    return attn_output, None
                    
            else:
                if seq_len - window_size <0:
                    return self.attention._attn(query, key, value, attention_mask, head_mask)
                if "flash_attn" in self.config.attn_type or self.config.attn_type == "step_attn":
                    combined_key = key[:,:, seq_len - window_size:, :]
                    combined_value = value[:,:, seq_len - window_size:, :]
                    combined_attention_mask = attention_mask[:,:,:,seq_len - window_size:]
                    if self.is_global_attn:
                        global_key, global_value, global_attention_mask = self.get_global_tokens(key, value, seq_len - 1)
                        combined_key = torch.cat([global_key, combined_key], dim = 2)
                        combined_value = torch.cat([global_value, combined_value], dim = 2)
                        combined_attention_mask = torch.cat([global_attention_mask, combined_attention_mask], dim = -1)
                    return self.attention._attn(query.half(), combined_key.half(), combined_value, combined_attention_mask, head_mask)
    
    def mqa_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        query = query.view(batch_size, num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size, 1, key_length, attn_head_size)
        # key = key.view(batch_size, num_attention_head, key_length, attn_head_size)
        # attn_scores = torch.zeros(
        #     batch_size * 1,
        #     query_length,
        #     key_length,
        #     dtype=query.dtype,
        #     device=key.device,
        # )
        # attn_scores = torch.baddbmm(
        #     attn_scores,
        #     query,
        #     key.transpose(1, 2),
        #     beta=1.0,
        #     alpha=(torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor),
        # )
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / self.norm_factor

        # print(query.shape, key.shape, attn_scores.shape)

        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


    def forward(self,
        hidden_states,
        attention_mask,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
        position_ids=None):

        has_layer_past = layer_past is not None
        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.attention.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.attention.num_attention_heads, 3 * self.attention.head_size)
        qkv = qkv.view(*new_qkv_shape)
        
        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.attention.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.attention.head_size : 2 * self.attention.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.attention.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.attention.rotary_ndims]
        query_pass = query[..., self.attention.rotary_ndims :]
        key_rot = key[..., : self.attention.rotary_ndims]
        key_pass = key[..., self.attention.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[-2]
            seq_len += offset
        cos, sin = self.attention.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids=position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output = self.long_attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        attn_output = self.attention._merge_heads(attn_output, self.attention.num_attention_heads, self.attention.head_size)

        attn_output = self.attention.dense(attn_output.type(self.dtype))
        
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (None,)

        return outputs