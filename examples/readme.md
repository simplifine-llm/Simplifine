This folder has examples on how to use Simplifine for fine-tuning.

The files included are as follows:

### clm_meas_example.py

This is the example used to re-produce the training speed using simplifine's DDP optimization. It can be run using cli. 

```bash
torchrun --nproc_per_node NUM_PROCESSORS example.py \
--hf_token HF_TOKEN \
--bs 1 \
--context_length 1024 \
--use_ddp 1
```

`NUM_PROCESSORS` would be the number of GPUs on the node to run this job. This runs a lora-clm job. replace `HF_TOKEN` with a huggingface token as this will use Llama-3-8b instruct's repo which is gated.