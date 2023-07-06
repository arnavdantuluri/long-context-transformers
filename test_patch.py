import torch
import transformers
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_apply_rotary_pos_emb
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import RotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from Mem_helpers.BPT.bpt_attention_plugin import BPTAttentionWrapperNeoX, BPTAttentionWrapperLLaMA
from Mem_helpers.BPT.bpt_feedforward_plugin import FeedForwardWrapperFalcon, FeedForwardWrapperNeoX, FeedForwardWrapperMPT, FeedForwardWrapperLLaMA
import os
from datetime import datetime

# model_path_or_name = "EleutherAI/pythia-1.4b"

model_path_or_name = "huggyllama/llama-7b"
# model_path_or_name = "EleutherAI/pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path_or_name, trust_remote_code=True)

factor = 1
max_positions = 2048 * factor
tokenizer.pad_token = tokenizer.mask_token
model.config.max_position_embeddings=max_positions
tokenizer.model_max_length = max_positions
#tokenizer.model_max_length = max_positions

if "neox" in model_path_or_name or "pythia" in model_path_or_name:
    for each in model.gpt_neox.layers:
        original_emb = each.attention.rotary_emb
        #base = torch.sqrt(1/each.attention.rotary_emb.inv_freq[1]).cpu()
        each.attention.rotary_emb = RotaryEmbedding(each.attention.rotary_ndims,max_positions,10000)
        each.attention.bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                    1, 1, max_positions, max_positions
                )
        each.attention = BPTAttentionWrapperNeoX(each.attention, query_chunk_size=512, key_chunk_size=512)
        each.mlp = FeedForwardWrapperNeoX(each.mlp, chunk_size=256, triton=True)

# MPT uses flash attention by default, no need for attention mechanisms dropped in here
elif "mpt" in model_path_or_name:
    for each in model.transformer.blocks:
        # each.attention = BPTAttentionWrapperNeoX(each.attention, query_chunk_size=512, key_chunk_size=512)
        each.ffn = FeedForwardWrapperMPT(each.ffn, chunk_size=128, triton=True)

elif "falcon" in model_path_or_name:
    for each in model.transformer.layers:
        original_emb = each.attention.rotary_emb
        #base = torch.sqrt(1/each.attention.rotary_emb.inv_freq[1]).cpu()
        each.attention.rotary_emb = RotaryEmbedding(each.attention.head_dim,max_positions,10000)
        each.mlp = FeedForwardWrapperFalcon(each.mlp, chunk_size=128, triton=True)

elif "llama" in model_path_or_name:
    for each in model.model.layers:
        #base = torch.sqrt(1/each.attention.rotary_emb.inv_freq[1]).cpu()
        each.self_attn.rotary_emb = LlamaRotaryEmbedding(each.self_attn.head_dim,max_positions,10000)
        each.self_attn = BPTAttentionWrapperLLaMA(each.self_attn, query_chunk_size=512, key_chunk_size=512)
        each.mlp = FeedForwardWrapperLLaMA(each.mlp, chunk_size=128, triton=False)

#torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
#Optimized 0:01:13.489796
torch.backends.cuda.sdp_kernel(enable_mem_efficient=True)
print(torch.backends.cuda.flash_sdp_enabled())
model = model.cuda().eval()

prompt = '''
Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who served as the 45th president of the United States from 2017 to 2021.

Trump graduated from the Wharton School of the University of Pennsylvania with a bachelor's degree in 1968. He became president of his father's real estate business in 1971 and renamed it The Trump Organization. He expanded the company's operations to building and renovating skyscrapers, hotels, casinos, and golf courses and later started side ventures, mostly by licensing his name. From 2004 to 2015, he co-produced and hosted the reality television series The Apprentice. Trump and his businesses have been involved in more than 4,000 state and federal legal actions, including six bankruptcies.

Trump's political positions have been described as populist, protectionist, isolationist, and nationalist. He won the 2016 United States presidential election as the Republican nominee against Democratic nominee Hillary Clinton despite losing the national popular vote.[a] He became the first U.S. president with no prior military or government service. His election and policies sparked numerous protests. The 2017–2019 special counsel investigation established that Russia interfered in the 2016 election to favor the election of Trump. Trump promoted conspiracy theories and made many false and misleading statements during his campaigns and presidency, to a degree unprecedented in American politics. Many of his comments and actions have been characterized as racially charged or racist, and many as misogynistic.

Trump ordered a travel ban on citizens from several Muslim-majority countries, diverted military funding towards building a wall on the U.S.–Mexico border, and implemented a policy of family separations for apprehended migrants. He rolled back more than 100 environmental policies and regulations in an aggressive attempt to weaken environmental protections. Trump signed the Tax Cuts and Jobs Act of 2017 which cut taxes for individuals and businesses and rescinded the individual health insurance mandate penalty of the Affordable Care Act. He appointed 54 federal appellate judges and three United States Supreme Court justices. Trump initiated a trade war with China and withdrew the U.S. from the proposed Trans-Pacific Partnership trade agreement, the Paris Agreement on climate change, and the Iran nuclear deal. Trump met with North Korean leader Kim Jong-un three times, but made no progress on denuclearization. He reacted slowly to the COVID-19 pandemic, ignored or contradicted many recommendations from health officials in his messaging, and promoted misinformation about unproven treatments and the need for testing.
''' * factor

start = datetime.now()
with torch.no_grad():
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    # we will use the below hacky util function to pass in the global mask, proper fix comes later, it needs to be a 2D list always, with batch size as the top level and the token index you want to attend to in the next
    #inputs['max_length'] = len(inputs[0]) + 512
    tokens = model.generate(inputs, use_cache = True, max_new_tokens = max_positions)
    outputs = tokenizer.decode(tokens[0])
print("Optimized", datetime.now() - start)