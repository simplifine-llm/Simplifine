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
from .train_engine_client import send_train_query, get_company_status, get_job_log, download_directory, stop_job
from .train_engine import sftPromptConfig, PromptConfig
from .url_class import url_config
import zipfile
import os
from dataclasses import asdict, fields
import warnings
from peft import LoraConfig
from dataclasses import dataclass
from trl import SFTConfig
from transformers import TrainingArguments

@dataclass
class wandbConfig:
    wandb_api_key:str
    project:str
    config:dict


class Client:
    def __init__(self, api_key:str='', gpu_type:str=''):
        self.api_key = api_key
        if api_key == '':
            raise Exception('API key not provided. Please provide your Simplifine API key.')
        self.gpu_type = gpu_type
        if gpu_type == 'l4':
            self.url = url_config.url_l4
        elif gpu_type == 'a100':
            self.url = url_config.url_a100
        else:
            print('GPU type not recognized. currently accepted versions are "a100" or "l4". Using default L4 server.')
            self.url = url_config.url
        self.wandb_config = None

    
    def clm_train_cloud(self, job_name:str,
        model_name:str, dataset_name:str=None, hf_token:str='', dataset_config_name:str=None, data_from_hf:bool=True,
    do_split:bool=True, split_ratio:float=0.2, use_peft:bool=False, lora_config:LoraConfig=None, 
    train_args:TrainingArguments=None, data:dict={}, wandb_config:wandbConfig=None, 
    use_ddp:bool=False, use_zero:bool=True, prompt_config:PromptConfig=None
    ):
        # client side checks
        if use_ddp and use_zero:
            raise ValueError("Only one dist method is accepted at once.")
        
        if use_ddp:
            train_args.deepspeed = None
        
        if train_args is None:
            raise ValueError('Training arguements must be provided')
        
        if data_from_hf and not dataset_name:
            raise ValueError('Dataset name must be provided if data is from Hugging Face')
        
        if use_ddp and train_args.gradient_checkpointing:
            print('[WARNING]: Gradient checkpointing is not supported with DDP. Disabling gradient checkpointing.')
            train_args.gradient_checkpointing = False
        
        if train_args.deepspeed is None and use_zero:
            print('[WARNING]: Zero is enabled but deepspeed is not.This will use default settings on Simplfiine servers.')
        
        
        # converting the 3 dataclasses, lora_config, sft_config, sft_prompt_config to dictionaries
        if lora_config is None:
            lora_config = LoraConfig()
        lora_config_dict = asdict(lora_config)
        train_args_dict = asdict(train_args)
        prompt_config = asdict(prompt_config)

        # making sure invalid fields are not passed to the server
        valid_fields = {f.name for f in fields(LoraConfig)}
        lora_config_dict = {k: v for k, v in lora_config_dict.items() if k in valid_fields}

        valid_fields = {f.name for f in fields(TrainingArguments)}
        train_args_dict = {k: v for k, v in train_args_dict.items() if k in valid_fields}

        valid_fields = {f.name for f in fields(PromptConfig)}
        prompt_config_dict = {k: v for k, v in prompt_config.items() if k in valid_fields}

        if wandb_config is not None:
            wandb_config_dict = asdict(wandb_config)
        else:
            wandb_config_dict = None

        config = {
                'api_key': self.api_key,
                'job_name': job_name,
                'type':'clm_v2',
                'model_name':model_name,
                'dataset_name': dataset_name,
                'args': {
                    'train_args': train_args_dict,
                    'lora_config': lora_config_dict,
                    'prompt_config': prompt_config_dict,
                    'data': data,
                    'data_from_hf': data_from_hf,
                    'hf_token': hf_token,
                    'dataset_config_name': dataset_config_name,
                    'do_split': do_split,
                    'split_ratio': split_ratio,
                    'use_peft': use_peft,
                    'wandb_config': wandb_config_dict,
                    'use_ddp': use_ddp,
                    'use_zero': use_zero
        }
        }

        send_train_query(config, url=self.url)


    def sft_train_cloud(self, job_name:str,
        model_name:str, dataset_name:str=None, hf_token:str='', dataset_config_name:str=None, data_from_hf:bool=True,
        do_split:bool=True, split_ratio:float=0.2, use_peft:bool=False, lora_config:LoraConfig=None, 
        sft_config:SFTConfig=None, data:dict={}, wandb_config:wandbConfig=None, 
        use_ddp:bool=False, use_zero:bool=True, sft_prompt_config:sftPromptConfig=None):

        # client side checks
        if use_ddp and use_zero:
            raise ValueError("Only one dist method is accepted at once.")
        
        if use_ddp:
            sft_config.deepspeed = None
        
        if sft_prompt_config.response_template not in sft_prompt_config.template:
            raise ValueError('The response template must be in the template')

        if sft_prompt_config.system_message_key and sft_prompt_config.system_message:
            raise ValueError('Only provide key from dataset or system message as a string, not both')
        
        if sft_config is None:
            raise ValueError('SFT config must be provided')
        
        if data_from_hf and not dataset_name:
            raise ValueError('Dataset name must be provided if data is from Hugging Face')
        
        if sft_config.report_to == 'wandb' and wandb_config is None:
            raise ValueError('Wandb config must be provided if report_to is wandb')

        if sft_config.deepspeed is None and use_zero:
            print('[WARNING]: Zero is enabled but deepspeed is not.This will use default settings on Simplfiine servers.')
        
        if use_ddp and sft_config.gradient_checkpointing:
            print('[WARNING]: Gradient checkpointing is not supported with DDP. Disabling gradient checkpointing.')
            sft_config.gradient_checkpointing = False
        
        # converting the 3 dataclasses, lora_config, sft_config, sft_prompt_config to dictionaries
        if lora_config is None:
            lora_config = LoraConfig()
        lora_config_dict = asdict(lora_config)
        sft_config_dict = asdict(sft_config)
        sft_prompt_config_dict = asdict(sft_prompt_config)

        # making sure invalid fields are not passed to the server
        valid_fields = {f.name for f in fields(LoraConfig)}
        lora_config_dict = {k: v for k, v in lora_config_dict.items() if k in valid_fields}

        valid_fields = {f.name for f in fields(SFTConfig)}
        sft_config_dict = {k: v for k, v in sft_config_dict.items() if k in valid_fields}

        valid_fields = {f.name for f in fields(sftPromptConfig)}
        sft_prompt_config_dict = {k: v for k, v in sft_prompt_config_dict.items() if k in valid_fields}

        if wandb_config is not None:
            wandb_config_dict = asdict(wandb_config)
        else:
            wandb_config_dict = None

        config = {
                'api_key': self.api_key,
                'job_name': job_name,
                'type':'sft_v2',
                'model_name':model_name,
                'dataset_name': dataset_name,
                'args': {
                    'sft_config': sft_config_dict,
                    'lora_config': lora_config_dict,
                    'sft_prompt_config': sft_prompt_config_dict,
                    'data': data,
                    'data_from_hf': data_from_hf,
                    'hf_token': hf_token,
                    'dataset_config_name': dataset_config_name,
                    'do_split': do_split,
                    'split_ratio': split_ratio,
                    'use_peft': use_peft,
                    'wandb_config': wandb_config_dict,
                    'use_ddp': use_ddp,
                    'use_zero': use_zero
        }
        }

        send_train_query(config, url=self.url)


    def get_all_jobs(self):
        """
        Function to get the status of a job.
        This function returns all of the jobs under your API key.
        """
        company_job_data = get_company_status(api_key=self.api_key, url=self.url)
        return company_job_data

    def get_status_id(self, job_id:str=''):
        all_jobs = get_company_status(self.api_key, url=self.url)['response']
        for i in all_jobs:
            if i['job_id'] == job_id:
                return i['status']
        raise Exception('Job ID not found')

    def get_train_logs(self, job_id:str=''):
        return get_job_log(self.api_key, job_id, url=self.url)

    def unzip_model(self, zip_path, extract_to):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Model unzipped successfully to {extract_to}")
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is not a valid zip file.")

    # unzip the model and then delete the zip file
    def download_model(self, job_id: str, extract_to: str):
        """
        function to download the model from Simplfine cloud, unzip it, and delete the zip file
        args:
            api_key: your simplifine API key/token
            job_id: the job id of the model to download
            extract_to: the path to extract the model. this should be a directory.
        """
        zip_name = f"{job_id}.zip"
        zip_path = os.path.join(extract_to, zip_name) 
        download_directory(self.api_key, job_id, zip_path, url=self.url)
        self.unzip_model(zip_path, extract_to)
        try:
            os.remove(zip_path)
            print(f"Deleted the zip file at {zip_path}")
        except Exception as e:
            print(f"Error deleting the zip file: {e}")
        print('Model downloaded, unzipped, and zip file deleted successfully!')
    
    def stop_job(self, job_id:str=''):
        if job_id == '':
            raise Exception('Please provide a job ID to stop the job.')
        return stop_job(self.api_key, job_id, url=self.url)