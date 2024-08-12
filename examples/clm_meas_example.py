from simplifine_alpha.train_engine import clm_train, PromptConfig
import transformers as ts
import datasets as ds
import os
from peft import LoraConfig
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--hf_token", type=str, default="")
argparser.add_argument("--bs", type=int, default=1, help="Batch size")
argparser.add_argument("--context_length", type=int, default=512, help="Context length")
argparser.add_argument("--use_ddp", type=int, default=0, help="Whether to use ddp, 1 is use, 0 is not use")

def clm_meas(hf_token:str='', bs:int=1, context_length:int=512, use_ddp:bool=False):
    # processing C4 demo dataset
    script_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(script_path, 'c4_demo.json')
    raw_dataset = ds.Dataset.from_json(dataset_path)
    data = {'text': raw_dataset['text']}
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    args = ts.TrainingArguments(
        output_dir="output",
        overwrite_output_dir=True,
        per_device_train_batch_size=bs,
        num_train_epochs=1,
        logging_dir="logs",
        bf16=True,
        report_to="none",
    )
    target_modules = ["q_proj", "k_proj", "v_proj"]
    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    prompt_config = PromptConfig(context_length=context_length)
    clm_train(
    model_name=model_id, hf_token=hf_token, data_from_hf=False,
    do_split=False, split_ratio=0.2, use_peft=True, lora_config=peft_config, 
    train_args=args, data=data, use_ddp=use_ddp, prompt_config=prompt_config
    )

if __name__ == "__main__":
    args = argparser.parse_args()
    hf_token = args.hf_token
    bs = args.bs
    context_length = args.context_length
    use_ddp_arg = args.use_ddp
    
    if use_ddp_arg == 1:
        use_ddp = True
    else:
        use_ddp = False

    clm_meas(hf_token=hf_token, bs=bs, context_length=context_length, use_ddp=use_ddp)