## long-context-transformers
A repository to get train transformers to access longer context for causal language models, most of these methods are still in testing. Try them out if you'd like but please lmk your results so we don't duplicate work :)
Exploring finetuning public checkpoints on filtered datasets to extend range of pre-trained models a la MPT-7B

## Currently supported 
Currently has code for Flash Attention + QLoRa, tested to work with NeoX models

Also has code for patching NeoX models with Blockwise Parallel Transformer attention (able to support 42k tokens on 160m model with single A100 gpu)

Will setup longformer and landmarks soon

## Training examples WIP
```bash

```
## Multiple GPUS
multiple gpus should be supported with 🤗 accelerate since QLoRa uses that but I have not tested it yet
