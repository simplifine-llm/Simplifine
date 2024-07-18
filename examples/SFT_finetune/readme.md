This folder ontains an example on instruction-tuning an LLM with supervised fine-tuning (SFT).

In this example, you will see how to design a format, and what columns from a huggingface dataset to use to fine-tune this model.

There are 2 methods of distributed training supported so far, ZeRO (DeepSpeed) and torch DDP. 

ZeRO enables training larger models on cheaper infra, by sharding gradients, parameters and optimizer states across processes. 

DDP, would require a replica on each worker (GPU) and as such, if the model cannot fit on a single GPU, would not work. 

To run the model, use torchrun such as this: 

```sh
torchrun --nrpoc_per_node NUM_WORKERS sft_ft.py
```

Note that NUM_WORKERS is the number of GPUs in a single node. 

This requires for you to have installed the relevant NVCC and CUDA libraries. 
