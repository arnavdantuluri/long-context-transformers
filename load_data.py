# from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# dataset_name = "timdettmers/openassistant-guanaco"
# save_path = "/p/scratch/ccstdl/dantuluri1/arxiv-summarization"
# dataset = load_dataset(dataset_name)
# print(dataset)
# dataset.save_to_disk(save_path)
# dataset = load_from_disk(save_path)
# print(dataset)

model_name = "huggyllama/llama-7b"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)