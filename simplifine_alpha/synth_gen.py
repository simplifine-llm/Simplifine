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
from openai import OpenAI, AsyncOpenAI
from typing import List, Union
import asyncio
import PyPDF2
import os
import json
import uuid
from logger import *
import pandas as pd
from tabulate import tabulate
import re

class openAI_job_manager:
    def __init__(self, api_key:str='', default_model='gpt-4o-mini'):
        self.client = OpenAI(api_key=api_key)
        self.client_async = AsyncOpenAI(api_key=api_key)
        self.model = default_model
        self.script_path = os.path.dirname(os.path.abspath(__file__))
    

    def get_batch_jobs(self):
        logger_path = os.path.join(self.script_path, 'batch_requests', 'logger.jsonl')
        all_meta = read_jsonl(logger_path)
        return all_meta
    

    def get_batch_status(self, batch_id):
        return get_batch_status(self.client, batch_id)
    

    def get_batch_type(self, batch_id):
        return get_batch_type(batch_id)
    

    def get_batch_response(self, batch_id):
        return retireve_batch_results(self.client, batch_id)
    

    def plot_batch_table(self, drop_columns:Union[List[str], str] = []):

        if isinstance(drop_columns, str):
            drop_columns = [drop_columns]
        
        logger_path = os.path.join(self.script_path, 'batch_requests', 'logger.jsonl')
        df = pd.read_json(logger_path, lines=True)

        if drop_columns:
            df = df.drop(columns=drop_columns)

        print(tabulate(df, headers='keys', tablefmt='pretty'))


    def generate(self, messages, model=None):
        if model is None:
            model = self.model
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content
    


class Generator(openAI_job_manager):
    def __init__(self, api_key:str='', default_model='gpt-4o-mini'):
        
        super().__init__(api_key=api_key, default_model=default_model)

        self.default_rational_system_prompt = """You are a helpful assistant. 
        You are tasked with creating rationale for why an answer is appropriate.
        This is based on the question and answer provided by the user. Just provide the rationale for the answer.
        Do not say anything other than the rationale.
        An example is below:
        User: question: what is the capital of France?, Answer: Paris.
        Assistant: Paris is the capital of France. It is a well-known fact that Paris is the capital of France."""

        self.default_question_system_prompt = """You are a helpful assistant.
        You are tasked with creating a question based on a prompt provided by the user.
        Just provide the question based on the prompt.
        Do not say anything other than the question.
        An example is below:
        User: Paris is the capital of France.
        Assistant: What is the capital of France?"""


    def generate_rational(self, questions:List[str], answers:List[str],
                          system_prompt:str=None, model:str=None):
        """
        Generate rationale for a list of questions and answers.
        Note that this function is synchronous and will be expensive for large lists.
        Use the generate_rational_async function for large lists.
        """
        if model is None:
            model = self.model
        if system_prompt is None:
            system_prompt = self.default_rational_system_prompt
        responses = []
        for q,a,num in zip(questions, answers, range(len(questions))):
            message = [
                {"role": "system", "content": system_prompt},
                {'role': 'user', 'content': f'question: {q}, Answer: {a}'}
            ]
            response = self.generate(message, self.model)
            responses.append(response)
        return responses


    def generate_rational_batch_file(self, questions: List[str], answers: List[str], max_tokens: int = 1000,
                                    system_prompt: str = None, model: str = None):
        """
        Generate a JSONL file for a batch request to OpenAI for rationale generation 
        based on a list of questions and answers.
        """
        if model is None:
            model = self.model
        if system_prompt is None:
            system_prompt = self.default_rational_system_prompt

        batch_requests = []

        for i, (q, a) in enumerate(zip(questions, answers)):
            # Create the message structure as per OpenAI's expected format
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'question: {q}, Answer: {a}'}
            ]
            # Append the data to batch_requests with the required structure
            request = {
                "custom_id": f"request-{i+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": message,
                    "max_tokens": max_tokens
                }
            }
            batch_requests.append(request)

        # Define the output file path
        file_name = str(uuid.uuid4()) + '.jsonl'

        # check if the folder batch_requests exists
        if not os.path.exists(f"{self.script_path}/batch_requests"):
            os.makedirs(f"{self.script_path}/batch_requests")
        
        output_file = f"{self.script_path}/batch_requests/{file_name}"

        # Write the JSONL file
        with open(output_file, 'w') as f:
            for request in batch_requests:
                json.dump(request, f)
                f.write('\n')

        return output_file, file_name
    

    def generate_batch_rational(self, questions: List[str], answers: List[str], max_tokens: int = 1000,
                                system_prompt: str = None, model: str = None, description: str = 'OpenAI Batch Request - Simplifine'):
        """
        Generate rationale for a list of questions and answers.
        This function uses openAI batches API and will be cheaper.
        """

        if len(questions) != len(answers):
            raise ValueError("The number of questions and answers must be equal.")

        output_file, file_name = self.generate_rational_batch_file(questions, answers, max_tokens, system_prompt, model)
        batch_input_file = self.client.files.create(
        file=open(output_file, "rb"),
        purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        returned_json = self.client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description":description
        }
        )

        log_batch_metadata(returned_json.id, batch_input_file_id, file_name, description,task='GEN-RATIONAL',task_specific_paramters={})


    def generate_questions(self, passages:List[str], system_prompt:str=None, model:str=None):
        if model is None:
            model = self.model
        if system_prompt is None:
            system_prompt = self.default_question_system_prompt
        responses = []
        for p in passages:
            message = [
                {"role": "system", "content": system_prompt},
                {'role': 'user', 'content': f'{p}'}
            ]
            response = self.generate(message, self.model)
            responses.append(response)
        return responses
    

    def generate_questions_batch_file(self, passages: List[str], max_tokens: int = 1000,
                                    system_prompt: str = None, model: str = None):
        """
        Generate a JSONL file for a batch request to OpenAI for questions generation 
        based on a list of passages.
        """
        if model is None:
            model = self.model
        if system_prompt is None:
            system_prompt = self.default_question_system_prompt

        batch_requests = []

        for i, p in enumerate(passages):
            # Create the message structure as per OpenAI's expected format
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'{p}'}
            ]
            # Append the data to batch_requests with the required structure
            request = {
                "custom_id": f"request-{i+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": message,
                    "max_tokens": max_tokens
                }
            }
            batch_requests.append(request)

        # Define the output file path
        file_name = str(uuid.uuid4()) + '.jsonl'

        # check if the folder batch_requests exists
        if not os.path.exists(f"{self.script_path}/batch_requests"):
            os.makedirs(f"{self.script_path}/batch_requests")
        
        output_file = f"{self.script_path}/batch_requests/{file_name}"

        # Write the JSONL file
        with open(output_file, 'w') as f:
            for request in batch_requests:
                json.dump(request, f)
                f.write('\n')

        return output_file, file_name


    def generate_batch_questions(self, passages: List[str], max_tokens: int = 1000,
                                system_prompt: str = None, model: str = None, description: str = 'OpenAI Batch Request - Simplifine'):
        """
        Generate rationale for a list of questions and answers.
        This function uses openAI batches API and will be cheaper.
        """

        output_file, file_name = self.generate_questions_batch_file(passages, max_tokens, system_prompt, model)
        batch_input_file = self.client.files.create(
        file=open(output_file, "rb"),
        purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        returned_json = self.client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description":description
        }
        )

        log_batch_metadata(returned_json.id, batch_input_file_id, file_name, description,task='GEN-QUESTION',task_specific_paramters={})
    

    def dataset_generation_batch(self, batch_id):
        """
        Generate the dataset based on the batch_id.
        """
        batch_status = self.get_batch_status(batch_id)
        if batch_status != 'completed':
            raise ValueError("The batch request is not completed yet.")
        batch_type = self.get_batch_type(batch_id)
        
        matched_response = match_response_to_request(batch_id, self.client)
        if batch_type == 'GEN-RATIONAL':
            output = {'questions': [], 'answers': [], 'rationales': []}
            pattern = r"question:\s*(.*?),\s*answer:\s*(.*)"
            # Use re.search to find the pattern in the text
            for req, resp in zip(matched_response['request'], matched_response['response']):
                match = re.search(pattern, req, re.IGNORECASE)
                if match:
                    question = match.group(1).strip()
                    answer = match.group(2).strip()
                    output['questions'].append(question)
                    output['answers'].append(answer)
                    output['rationales'].append(resp)
                else:
                    print[f'[WARNING] The pattern was not found in the following request {req}']
                    continue
            return output
        elif batch_type == 'GEN-QUESTION':
            return matched_response
        else:
            raise ValueError("The batch type is not recognized.")


    def read_pdf(self, document_path):
        with open(document_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
            text = ''
            for page in range(reader.numPages):
                text += reader.getPage(page).extractText()
        return text