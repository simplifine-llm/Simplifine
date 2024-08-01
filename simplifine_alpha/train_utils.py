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
import zipfile
import os
from .url_class import url_config
from dataclasses import asdict, fields
import warnings
from peft import LoraConfig


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


    def cls_train_cloud(self, job_name:str='', dataset_name:str='', model_name:str='', 
                        inputs:list=[], labels:list=[], from_hf:bool=True, use_peft:bool=False,
                        peft_config=None, hf_token:str='', lr:float=5e-5, num_epochs:int=3,
                        batch_size:int=8, use_fp16:bool=False, use_bf16:bool=False,
                        use_ddp:bool=False, use_zero:bool=True, use_gradient_checkpointing:bool=False,
                        report_to:str='none', wandb_api_key:str='', do_split:bool=True, split_ratio:float=0.2,
                        gradient_accumulation_steps:int=4, hf_column_input:str='', hf_column_label:str='', lr_scheduler_type:str='linear'):
        pass

    def clm_train_cloud(self, job_name:str='',
                        model_name:str='', dataset_name:str="",
                        context_length:int=128, data:list=[],
                        num_epochs:int=3, batch_size:int=8, use_fp16:bool=False, use_bf16:bool=False,
                        lr:float=5e-5, from_hf:bool=True, do_split:bool=True, split_ratio:float=0.2,
                        gradient_accumulation_steps:int=4, use_gradient_checkpointing:bool=False,
                        report_to:str='none', wandb_api_key:str='', eval_accumulation_steps:int=4,
                        use_peft:bool=False, peft_config=None, hf_token:str='',
                        hf_column:str='text', lr_scheduler_type:str='linear',
                        use_ddp:bool=False, use_zero:bool=True):
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
            hf_token (str): The huggingface token.
            report_to (str): The reporting destination.
            deepspeed_config (dict): The deepspeed configuration.
            gradient_accumulation_steps (int): The number of gradient accumulation steps.
        """

        # client side checks
        if use_zero and use_ddp:
            raise Exception('Zero and DDP cannot be used together. Please choose one.')
        if use_zero and use_peft:
            raise Exception('Zero and PEFT cannot be used together. Please choose one. We are working hard on this')
        if use_ddp and use_gradient_checkpointing:
            raise Exception('DDP and gradient checkpointing cannot be used together. Please choose one.')

        # check for peft and peft config
        if use_peft:
            if peft_config is None:
                warnings.warn('PEFT is enabled but no PEFT config provided. Using default mode.')
                _peft_config = None
            else:
                _peft_config = asdict(peft_config)
                # Convert target_modules to a list if it is a set
                if 'target_modules' in _peft_config and isinstance(_peft_config['target_modules'], set):
                    _peft_config['target_modules'] = list(_peft_config['target_modules'])
                
                # Get the valid fields of LoraConfig
                valid_fields = {f.name for f in fields(LoraConfig)}

                # Filter the dictionary to include only valid fields
                _peft_config = {k: v for k, v in _peft_config.items() if k in valid_fields}

                # Debug print statement
                print(f"PEFT config: {_peft_config}")


        
        config = {
            'api_key': self.api_key,
            'job_name': job_name,
            'type':'clm',
            'model_name':model_name,
            'dataset_name': dataset_name,
            'args': {
                'data':data,
                'lr': lr,
                'context_length': context_length,
                'use_peft': use_peft,
                'peft_config': _peft_config,
                'num_train_epochs': num_epochs,
                'batch_size': batch_size,
                'use_ddp': use_ddp,
                'use_zero': use_zero,
                'use_fp16': use_fp16,
                'use_bf16':use_bf16,
                'use_gradient_checkpointing': use_gradient_checkpointing,
                'huggingface_token': hf_token,
                'report_to': report_to,
                'wandb_api_key': wandb_api_key,
                'do_split': do_split,
                'split_ratio': split_ratio,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'per_device_batch_size': batch_size,
                'from_hf': from_hf,
                'hf_column': hf_column,
                'lr_scheduler_type': lr_scheduler_type,
                'eval_accumulation_steps': eval_accumulation_steps
            }
        }
        send_train_query(config, url=self.url)

    def sft_train_cloud(self, job_name:str='',
                        model_name:str='', dataset_name:str='', data:dict={},
                        keys:str='', template:str='', response_template:str='',
                        learning_rate:float=1e-5, num_train_epochs:int=1,  per_device_batch_size:int=1,
                        batch_size:int=1, use_ddp:bool=False, eval_accumulation_steps:int=4,
                        use_fp16:bool=False, use_deepspeed:bool=False, use_peft:bool=False,
                        use_gradient_checkpointing:bool=False, use_activation_checkpointing:bool=False,
                        huggingface_token:str='', do_split:bool=True, split_ratio:float=0.2,
                        prompt_template:str='', report_to:str='none', max_seq_length:int=2048,
                        deepspeed_config:dict={}, gradient_accumulation_steps:int=1, 
                        from_hf:bool=True, peft_config=None, use_zero:bool=False,
                        ):
        """
        Function to send the training query to the server. This is for supervised fine-tuning. 
        args:
            api_key: your simplifine API key/token
            job_name: the name of the job
            model_name: the name of the model to train. At this point, the only models accepted are from huggingface.
            dataset_name: the name of the dataset to use
            keys: the keys to use for the template
            template: the template to use for the training
            learning_rate: the learning rate for the training
            num_train_epochs: the number of training epochs
            batch_size: the batch size for training
            use_ddp: whether to use distributed data parallelism
            use_fp16: whether to use 16-bit floating point precision
            use_deepspeed: whether to use deepspeed
            use_peft: whether to use PEFT
            use_gradient_checkpointing: whether to use gradient checkpointing

        """

        # client side checks
        if use_zero and use_ddp:
            raise Exception('Zero and DDP cannot be used together. Please choose one.')
        if use_zero and use_peft:
            raise Exception('Zero and PEFT cannot be used together. Please choose one. We are working hard on this')
        if use_ddp and use_gradient_checkpointing:
            raise Exception('DDP and gradient checkpointing cannot be used together. Please choose one.')
        # SFT specfic checks
        if response_template == '':
            raise Exception('Please provide a response template for the model to train.')
        if template == '':
            raise Exception('Please provide a prompt template for the model to train.')
        if response_template not in template:
            raise Exception('Response template not found in the main template.')
    
        # check for peft and peft config
        if use_peft:
            if peft_config is None:
                warnings.warn('PEFT is enabled but no PEFT config provided. Using default mode.')
                _peft_config = None
            else:
                _peft_config = asdict(peft_config)
                # Convert target_modules to a list if it is a set
                if 'target_modules' in _peft_config and isinstance(_peft_config['target_modules'], set):
                    _peft_config['target_modules'] = list(_peft_config['target_modules'])
                
                # Get the valid fields of LoraConfig
                valid_fields = {f.name for f in fields(LoraConfig)}

                # Filter the dictionary to include only valid fields
                _peft_config = {k: v for k, v in _peft_config.items() if k in valid_fields}

                # Debug print statement
                print(f"PEFT config: {_peft_config}")
        else:
            _peft_config = None
            
            
        config = {
            'api_key': self.api_key,
            'job_name': job_name,
            'type':'sft',
            'model_name':model_name,
            'dataset_name': dataset_name,
            'args': {
                'use_peft': use_peft,
                'peft_config': _peft_config,
                'lr': learning_rate,
                'keys': keys,
                'template': template,
                'num_train_epochs': num_train_epochs,
                'batch_size': batch_size,
                'use_ddp': use_ddp,
                'use_zero': use_zero,
                'use_fp16': use_fp16,
                'use_deepspeed': use_deepspeed,
                'do_split': do_split,
                'split_ratio': split_ratio,
                'use_gradient_checkpointing': use_gradient_checkpointing,
                'use_activation_checkpointing': use_activation_checkpointing,
                'huggingface_token': huggingface_token,
                'response_template': response_template,
                'prompt_template': prompt_template,
                'max_seq_length': max_seq_length,
                'report_to': report_to,
                'deepspeed_config': deepspeed_config,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'per_device_batch_size': per_device_batch_size,
                'from_hf': from_hf,
                'data': data,
                'eval_accumulation_steps':eval_accumulation_steps
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