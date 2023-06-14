import torch
from datasets import load_dataset
from transformers.models.gpt_neox.modeling_gpt_neox import RotaryEmbedding
from transformers.trainer_utils import get_last_checkpoint
from itertools import chain
from typing import Optional
from dataclasses import dataclass, field
from longformer import LongformerAttentionWrapperWithRotary, set_global_attention_indices
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import zstandard
import evaluate

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import RotaryEmbedding
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model 

model_path_or_name = "EleutherAI/pythia-160m"
config = AutoConfig.from_pretrained(model_path_or_name)
config.attention_window = [2048 for _ in range(config.num_hidden_layers)]
config.dtype = torch.float16
config.is_global_attn = True
config.max_global_tokens = 64
config.attention_probs_dropout_prob = 0
config.attn_type = "step_attn"

tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, use_cache=False, use_local_files=True)
model = AutoModelForCausalLM.from_pretrained(model_path_or_name, use_cache=False, use_local_files=True).half().eval().cuda()

max_positions = 8192
tokenizer.pad_token = tokenizer.mask_token
model.config.max_position_embeddings=max_positions
tokenizer.model_max_length = max_positions

layer_id = 0
for each in model.gpt_neox.layers:
    original_emb = each.attention.rotary_emb
    each.attention.rotary_emb = RotaryEmbedding(each.attention.rotary_ndims,max_positions,10000).cuda()
    each.attention.bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ).cuda()
    each.attention = LongformerAttentionWrapperWithRotary(each.attention, max_seqlen = max_positions, config = config, layer_id = layer_id).cuda()

for param in model.parameters():
    param.requires_grad = False

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

model = model.half().cuda().eval()
set_global_attention_indices(model, [[i for i in range(10)] + [i for i in range(65, 70)]] * 8) #8 is batch size for training loop

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-160m",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    max_positions: Optional[int] = field(
        default=4096,
        metadata={
            "help": (
                "The maximun sequence length of the model."
            )
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="pile", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

def main(model):
    model_args, data_args, training_args = ModelArguments(), DataTrainingArguments(), TrainingArguments("./", per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adafactor",
)
    training_args.max_steps = 1000
    training_args = training_args.set_testing(batch_size=4)
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.mask_token
    max_positions = model_args.max_positions
    tokenizer.model_max_length = max_positions

    # patching for the random contiguous tensors bug
    for p in model.parameters():
        p = p.contiguous()

    def merge_questions_and_answers(examples):
        out = tokenizer([question + " " + answer for question, answer in zip(examples["input"], examples["output"])])
        return out

    block_size = tokenizer.model_max_length
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    

    if data_args.dataset_name == "pile":
        train_path = "https://the-eye.eu/public/AI/pile/train/01.jsonl.zst"
        val_path = "https://the-eye.eu/public/AI/pile/val.jsonl.zst"

        data_files = {
            "train": train_path,
            "validation": val_path,
        }
        datasets = load_dataset("json", data_files=data_files, streaming=True)
        datasets = datasets.filter(lambda x: len(x["text"])>=max_positions)
        tokenized_datasets = datasets.map(
            lambda examples: tokenizer(examples["text"]),
            batched=True,
        )
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )
        lm_datasets = lm_datasets.filter(lambda x: len(x["input_ids"])>=max_positions)
    elif data_args.dataset_name == "qasper":
        datasets = load_dataset("tau/scrolls", "qasper")
        datasets.pop("test")
        tokenized_datasets = datasets.map(
           merge_questions_and_answers,
           batched=True,
           num_proc = 1,
           remove_columns = datasets["train"].column_names,
           desc="Running tokenizer on dataset",
        )

        lm_datasets = tokenized_datasets.map(
           group_texts,
           batched=True,
           num_proc=1,
           desc=f"Grouping texts in chunks of {block_size}",
        )
    else:
        raise Exception("Sorry, please the dataset specified can not be recognized")
    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= train_dataset.with_format("torch"),
        eval_dataset= eval_dataset.with_format("torch"),
        tokenizer = tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics)

    checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

main(model)