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

This is how simplifine is ran on our servers as well. If you wish to run jobs on our servers, take a look at `cloud_quick_start.ipynb`.

### cloud_quick_start.ipynb

This simple notebook demonstrates how to quickly switch between local setup to cloud-based setup. This would be just a simple change of syntax. This example demonstrates how the process of fine-tuning, in fp16 percision and using ZeRO, could be simplified using simplifine. If you are interested, you would require a simplfine API key to get started. We do provision free credits to get a feel for Simplifine's serverless API. **Get free Simplifine Cloud Credits to finetune [here](https://www.simplifine.com/api-key-interest)**