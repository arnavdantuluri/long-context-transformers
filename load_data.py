from datasets import load_dataset, load_from_disk

dataset_name = "timdettmers/openassistant-guanaco"
save_path = "/p/scratch/ccstdl/dantuluri1/arxiv-summarization"
# dataset = load_dataset(dataset_name)
# print(dataset)
# dataset.save_to_disk(save_path)
dataset = load_from_disk(save_path)
print(dataset)
# from transformers import AutoTokenizer 
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")