from datasets import load_dataset, load_from_disk

dataset = load_dataset("ccdv/arxiv-summarization")
print(dataset)
dataset.save_to_disk("/p/scratch/ccstdl/dantuluri1/arxiv-summarization")
dataset = load_from_disk("/p/scratch/ccstdl/dantuluri1/arxiv-summarization")
print(dataset)