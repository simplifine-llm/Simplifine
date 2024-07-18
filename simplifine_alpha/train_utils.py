'''
    Simplfine is an easy-to-use, open-source library for fine-tuning LLMs models quickly on your own hardware or cloud.
    Copyright (C) 2024  Simplifine Corp.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
# from train_engine import *
from train_engine_client import send_train_query
import uuid

def clm_train_cloud(api_key:str='', job_name:str='',
                    model_name:str='', dataset_name:str='', 
                    learning_rate:float=1e-5, num_train_epochs:int=1, 
                    batch_size:int=1, use_ddp:bool=False, 
                    context_length:int=1024, use_zero:bool=False,
                    use_fp16:bool=False, use_deepspeed:bool=False, use_peft:bool=False,
                    use_gradient_checkpointing:bool=False, use_activation_checkpointing:bool=False,
                    huggingface_token:str='', train_test_split_ratio:float=0.8,
                    prompt_template:str='', report_to:str='none',
                    deepspeed_config:dict={}, gradient_accumulation_steps:int=1, per_device_batch_size:int=1,
                    from_hf:bool=True, peft_config:dict={}):
    """
    Function to train a causal language model on the cloud.
    args:
        api_key (str): The API key for the cloud service.
        job_name (str): The name of the job.
        model_name (str): The name of the model to train.
        dataset_name (str): The name of the dataset to use.
        learning_rate (float): The learning rate for the training.
        num_train_epochs (int): The number of training epochs.
        batch_size (int): The batch size for training.
        use_ddp (bool): Whether to use distributed data parallelism.
        use_fp16 (bool): Whether to use 16-bit floating point precision.
        use_deepspeed (bool): Whether to use deepspeed.
        use_peft (bool): Whether to use PEFT.
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing.
        use_gactivation_checkpointing (bool): Whether to use activation checkpointing.
        huggingface_token (str): The huggingface token.
        response_template (str): The response template.
        prompt_template (str): The prompt template.
        report_to (str): The reporting destination.
        deepspeed_config (dict): The deepspeed configuration.
        gradient_accumulation_steps (int): The number of gradient accumulation steps.
    """

    job_id = str(uuid.uuid4())
    
    config = {
        'api_key': api_key,
        'job_id': job_id,
        'job_name': job_name,
        'type':'clm',
        'model_name':model_name,
        'dataset_name': dataset_name,
        'args': {
            'lr': learning_rate,
            'context_length': context_length,
            'use_peft': use_peft,
            'peft_config': peft_config,
            'num_train_epochs': num_train_epochs,
            'batch_size': batch_size,
            'use_ddp': use_ddp,
            'use_zero': use_zero,
            'use_fp16': use_fp16,
            'use_deepspeed': use_deepspeed,
            'use_gradient_checkpointing': use_gradient_checkpointing,
            'use_activation_checkpointing': use_activation_checkpointing,
            'huggingface_token': huggingface_token,
            'prompt_template': prompt_template,
            'report_to': report_to,
            'deepspeed_config': deepspeed_config,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'per_device_batch_size': per_device_batch_size,
            'from_hf': from_hf,
            'train_test_split_ratio':train_test_split_ratio,
            'output_dir':f'/tmp/{job_id}'
        }
    }
    send_train_query(config)

def sft_train_cloud(api_key:str='', job_name:str='',
                    model_name:str='', dataset_name:str='', 
                    keys:str='', template:str='', response_template:str='',
                    learning_rate:float=1e-5, num_train_epochs:int=1,  per_device_batch_size:int=1,
                    batch_size:int=1, use_ddp:bool=False,
                    use_fp16:bool=False, use_deepspeed:bool=False, use_peft:bool=False,
                    use_gradient_checkpointing:bool=False, use_activation_checkpointing:bool=False,
                    huggingface_token:str='',
                    prompt_template:str='', report_to:str='none',
                    deepspeed_config:dict={}, gradient_accumulation_steps:int=1, 
                    from_hf:bool=True, peft_config:dict={}, train_test_split_ratio:float=0.8, use_zero:bool=False,
                    ):
    job_id = str(uuid.uuid4())
    
    config = {
        'api_key': api_key,
        'job_id': job_id,
        'job_name': job_name,
        'type':'clm',
        'model_name':model_name,
        'dataset_name': dataset_name,
        'args': {
            'use_peft': use_peft,
            'peft_config': peft_config,
            'lr': learning_rate,
            'keys': keys,
            'template': template,
            'num_train_epochs': num_train_epochs,
            'batch_size': batch_size,
            'use_ddp': use_ddp,
            'use_zero': use_zero,
            'use_fp16': use_fp16,
            'use_deepspeed': use_deepspeed,
            'use_gradient_checkpointing': use_gradient_checkpointing,
            'use_activation_checkpointing': use_activation_checkpointing,
            'huggingface_token': huggingface_token,
            'response_template': response_template,
            'prompt_template': prompt_template,
            'report_to': report_to,
            'deepspeed_config': deepspeed_config,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'per_device_batch_size': per_device_batch_size,
            'from_hf': from_hf,
            'train_test_split_ratio':train_test_split_ratio,
            'output_dir':f'/tmp/{job_id}'
        }
    }
    send_train_query(config)