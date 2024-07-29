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

from openai import OpenAI
import os
import json
# from text_chunker import TextChunker
from tqdm import tqdm
from utils import chunk_text_by_words
from logger import *
import uuid
import re
import random
from openai import AsyncOpenAI




# TODO: add a function to check the status of a batch request
class synthetic:
    def __init__(self):
        self.client = OpenAI(
        api_key='',
        )
        self.async_client = AsyncOpenAI(
        api_key='',
        )
        self.model_name = 'gpt-4o'
        self.script_path = os.path.dirname(os.path.abspath(__file__))

        # check to see if file to store the batch requests exists
        if 'batch requests' not in os.listdir(self.script_path):
            print('Creating a directory to store batch requests')
            os.mkdir(os.path.join(self.script_path, 'batch requests'))
        
        # variable to store the current batch id
        self.current_batch = None
    

    def create_prompt_qa_gen(self, chunk:str):
        system_prompt = f"""
        You are a helpful assistant creating a question about the text. You are given a text and you need to create a question about it.
        Just respond with the qeusiton you would ask about the text. Say nothing else. 
        """
        user_prompt = f"""text: {chunk}"""

        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                ]
        return messages
    
    async def create_qa_pair_single_async(self, 
                              text:str, 
                              chunk_size:int=40, 
                              overlap:int=10,
                              shrink:bool=True,
                              ratio_of_shrink:float=0.5):
        """
        Create a synch call for QA generation. 
        **NOTE: this is a synchronous call, and it is not recommended to use it for large texts."""
        chunks = chunk_text_by_words(text, chunk_size, overlap)
        if shrink:
            # creating random indices to shrink the text
            shrink_indices = random.sample(range(len(chunks)), int(ratio_of_shrink*len(chunks)))
            shrunk_chunks = [chunks[i] for i in range(len(chunks)) if i in shrink_indices]

        questions, answers = [], []
        for chunk in tqdm(shrunk_chunks):
            answers.append(chunk)
            messages = self.create_prompt_qa_gen(chunk)
            completion = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages
            )
            print('\n----------------\n',completion.choices[0].message.content,'\n----------------\n')
            questions.append(completion.choices[0].message.content)
        return questions, answers, chunks, shrink_indices
    

    
    def create_qa_pair_single(self, 
                              text:str, 
                              chunk_size:int=40, 
                              overlap:int=10,
                              shrink:bool=True,
                              ratio_of_shrink:float=0.5):
        """
        Create a synch call for QA generation. 
        **NOTE: this is a synchronous call, and it is not recommended to use it for large texts."""
        chunks = chunk_text_by_words(text, chunk_size, overlap)
        if shrink:
            # creating random indices to shrink the text
            shrink_indices = random.sample(range(len(chunks)), int(ratio_of_shrink*len(chunks)))
            shrunk_chunks = [chunks[i] for i in range(len(chunks)) if i in shrink_indices]

        questions, answers = [], []
        for chunk in tqdm(shrunk_chunks):
            answers.append(chunk)
            messages = self.create_prompt_qa_gen(chunk)
            completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
            )
            questions.append(completion.choices[0].message.content)
        return questions, answers, chunks, shrink_indices
    
    def create_qa_pair_batch_file(self, text:str, file_name:str, chunk_size:int=200, overlap:int=10, shrink:bool=True, ratio_of_shrink:float=0.5):
        """
        Create a batch file for QA generation. 
        **NOTE: this is an asynchronous call, and it is recommended to use it for large texts."""
        chunks = chunk_text_by_words(text, chunk_size, overlap=overlap)
        shrink_indices = list(range(len(chunks)))
        if shrink:
            # creating random indices to shrink the text
            shrink_indices = random.sample(range(len(chunks)), int(ratio_of_shrink*len(chunks)))
            shrunk_chunks = [chunks[i] for i in range(len(chunks)) if i in shrink_indices]
        

        save_path = os.path.join(self.script_path, 'batch requests', file_name)
        batch_request_path = os.path.join(self.script_path, 'batch requests')
        if file_name in os.listdir(batch_request_path):
            print('File already exists')
            raise FileExistsError(f'File with name *{file_name}* already exists')

        count = 0
        for chunk in shrunk_chunks:
            reqID = f"request-{count}"
            count += 1
            batch_format = {
            "custom_id":"",
            "method":"",
            "url":"",
            "body":{}
            }
            messages = self.create_prompt_qa_gen(chunk)
            batch_format["custom_id"] = reqID
            batch_format["method"] = "POST"
            batch_format["url"] = "/v1/chat/completions"
            batch_format["body"]["model"] = self.model_name
            batch_format["body"]["messages"] = messages
            batch_format["body"]["max_tokens"] = 2000

            
            if os.path.exists(save_path):
                print('File exists')
                with open(save_path, 'a') as file:
                    for entry in [batch_format]:
                        file.write(json.dumps(entry) + '\n')

            else:
                print('File does not exists')
                existing_data = {}
                for col in ['custom_id', 'method', 'url', 'body']:
                    existing_data[col] = batch_format[col]
                existing_data = [existing_data]
                with open(save_path, 'w') as file:
                    for entry in existing_data:
                        file.write(json.dumps(entry) + '\n')
        
        return shrink_indices, chunks 
    

    # TODO: change to accomodate text as list (multiple docs)
    def QA_gen_batch(self, text:str, chunk_size:int=200, overlap:int=10, shrink:bool=True, ratio_of_shrink:float=0.5, description=""):
        """
        evaluating a batch of questions and answers.
        args:
        file_name: str, the name of the file containing the questions and answers
        returns:
        llm_answers: list, a list of answers from the model
        """
        file_name = str(uuid.uuid4()) + '.jsonl'
        shrink_indices, chunks  = self.create_qa_pair_batch_file(text, file_name, chunk_size, overlap, shrink, ratio_of_shrink)

        file_path = os.path.join(self.script_path, 'batch requests', file_name)
        batch_input_file = self.client.files.create(
        file=open(file_path, "rb"),
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

        task_specific_paramters = {'chunk_size':chunk_size, 'shrink_indices':shrink_indices, 'chunks':chunks}
        # save the metadata for the batch request
        log_batch_metadata(returned_json.id, batch_input_file_id, file_name, description, task_specific_paramters=task_specific_paramters, task='ANOT-QA')

    
    def create_QA_annotation_file(self, batch_id:str, file_path:str, overwrite:bool=False):
        """
        retireve the results of a batch request and save it as a jsonl file.
        **NOTE: file type accepted now is only .jsonl
        args:
        batch_id: str, the id of the batch request
        file_path: str, the path of the file to save the results

        """
        file_extension = file_path.split('.')[-1]
        if file_extension != 'jsonl':
            raise ValueError(f'File type is {file_extension}, only jsonl is supported at the moment.')
        # check if ID is for batch qa annotation
        path = os.path.join(script_path, 'batch requests', 'logger.jsonl')
        all_meta_data = read_jsonl(path)
        batch_not_found = True
        job_task_dont_match = True
        for i in range(len(all_meta_data)):
            if all_meta_data[i]['id'] == batch_id:
                batch_not_found = False
                if 'ANOT-QA' in all_meta_data[i]['task']:
                    job_task_dont_match = False
                break
        if batch_not_found:
            raise ValueError('Batch ID not found')
        if job_task_dont_match:
            raise ValueError(f'Batch ID {batch_id} does not match the task, task recorded as: {all_meta_data[i]["task"]}, target should be: ANOT-QA')
        matched_resp = match_response_to_request(batch_id, self.client)

        # writing to the path
        if os.path.exists(file_path):
            print(f'File {file_path} already exists')
            if overwrite:
                existing_data = {}
                for col in list(matched_resp.keys()):
                    existing_data[col] = matched_resp[col]
                existing_data = [existing_data]
                with open(file_path, 'w+') as file:
                    for entry in existing_data:
                        file.write(json.dumps(entry) + '\n')
            else:
                with open(file_path, 'a') as file:
                    for entry in [file_path]:
                        file.write(json.dumps(entry) + '\n')

        else:
            existing_data = {}
            for col in list(matched_resp.keys()):
                existing_data[col] = matched_resp[col]
            existing_data = [existing_data]
            with open(file_path, 'w+') as file:
                for entry in existing_data:
                    file.write(json.dumps(entry) + '\n')

    
    def create_prompt_summarization(self, text:str):
        system_prompt = f"""You are a helpful assistant summarizing a text. You are given a text and you need to create a summary of it.
        Be concise and to the point. Include the important information. 
        Just respond with the summary you would create for the text. Say nothing else. 
        """
        user_prompt = f"""text: {text}"""

        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                ]
        return messages

    def create_summarization_single(self, text:str):
        """
        Create a synch call for summarization. 
        **NOTE: this is a synchronous call, and it is not recommended to use it for large texts."""
        messages = self.create_prompt_summarization(text)
        completion = self.client.chat.completions.create(
        model=self.model_name,
        messages=messages
        )
        return completion.choices[0].message.content
    
    
    def create_summarization_batch_file(self, text_list:list, file_name:str):
        """
        Create a batch file for summarization. 
        **NOTE: this is an asynchronous call, and it is recommended to use it for large texts."""
        save_path = os.path.join(self.script_path, 'batch requests', file_name)
        batch_request_path = os.path.join(self.script_path, 'batch requests')
        if file_name in os.listdir(batch_request_path):
            print('File already exists')
            raise FileExistsError(f'File with name *{file_name}* already exists')
        
        count = 0
        for text in text_list:
            reqID = f"request-{count}"
            count += 1
            batch_format = {
            "custom_id":"",
            "method":"",
            "url":"",
            "body":{}
            }
            messages = self.create_prompt_summarization(text)
            batch_format["custom_id"] = reqID
            batch_format["method"] = "POST"
            batch_format["url"] = "/v1/chat/completions"
            batch_format["body"]["model"] = self.model_name
            batch_format["body"]["messages"] = messages
            batch_format["body"]["max_tokens"] = 2000

            
            if os.path.exists(save_path):
                print('File exists')
                with open(save_path, 'a') as file:
                    for entry in [batch_format]:
                        file.write(json.dumps(entry) + '\n')

            else:
                print('File does not exists')
                existing_data = {}
                for col in ['custom_id', 'method', 'url', 'body']:
                    existing_data[col] = batch_format[col]
                existing_data = [existing_data]
                with open(save_path, 'w') as file:
                    for entry in existing_data:
                        file.write(json.dumps(entry) + '\n')
    
    def summarization_gen_batch(self, text:list, description=""):
        """
        creating a batch of summarizations.
        args:
        text: list, a list of texts to summarize
        file_name: str, the name of the file containing the questions and answers
        description: str, the description of the batch request
        """
        file_name = str(uuid.uuid4()) + '.jsonl'
        self.create_summarization_batch_file(text, file_name)

        file_path = os.path.join(self.script_path, 'batch requests', file_name)
        batch_input_file = self.client.files.create(
        file=open(file_path, "rb"),
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

        # save the metadata for the batch request
        log_batch_metadata(returned_json.id, batch_input_file_id, file_name, description,task='ANOT-SUMMARIZATION',task_specific_paramters={})

    def create_summarization_annotation_file(self, batch_id:str, file_path:str, overwrite:bool=False):
        """
        retireve the results of a batch request and save it as a jsonl file.
        **NOTE: file type accepted now is only .jsonl
        args:
        batch_id: str, the id of the batch request
        file_path: str, the path of the file to save the results
        """
        file_extension = file_path.split('.')[-1]
        if file_extension != 'jsonl':
            raise ValueError(f'File type is {file_extension}, only jsonl is supported at the moment.')
        # check if ID is for batch qa annotation
        path = os.path.join(script_path, 'batch requests', 'logger.jsonl')
        all_meta_data = read_jsonl(path)
        batch_not_found = True
        job_task_dont_match = True
        for i in range(len(all_meta_data)):
            if all_meta_data[i]['id'] == batch_id:
                batch_not_found = False
                if 'ANOT-SUMMARIZATION' in all_meta_data[i]['task']:
                    job_task_dont_match = False
                break
        if batch_not_found:
            raise ValueError('Batch ID not found')
        if job_task_dont_match:
            raise ValueError(f'Batch ID {batch_id} does not match the task, task recorded as: {all_meta_data[i]["task"]}, target should be: ANOT-SUMMARIZATION')
        matched_resp = match_response_to_request(batch_id, self.client)

        # writing to the path
        if os.path.exists(file_path):
            print(f'File {file_path} already exists')
            if overwrite:
                existing_data = {}
                for col in list(matched_resp.keys()):
                    existing_data[col] = matched_resp[col]
                existing_data = [existing_data]
                with open(file_path, 'w+') as file:
                    for entry in existing_data:
                        file.write(json.dumps(entry) + '\n')
            else:
                with open(file_path, 'a') as file:
                    for entry in [file_path]:
                        file.write(json.dumps(entry) + '\n')

        else:
            existing_data = {}
            for col in list(matched_resp.keys()):
                existing_data[col] = matched_resp[col]
            existing_data = [existing_data]
            with open(file_path, 'w+') as file:
                for entry in existing_data:
                    file.write(json.dumps(entry) + '\n')
    

    def clf_prompt(self, text:str, classes:list=[], give_explanation:bool=False, example:str=""):
        if len(classes) == 0:
            raise ValueError('No classes provided')
        if give_explanation:
            exp_str = '''Provide a ratuionale for your classification. Your response should be formatted as: 
            #class: the assigned class, #rationale: the rational you used to get this label. 
            For example: User input: text: I love this movie, classes: Positive, Negative. your response: #class: Positive, #rationale: Positive because the user said they love the movie.'''
        else:
            exp_str = ''
        system_prompt = f"""You are a helpful assistant classifying a text. You are given a text and you need to classify it. The user will provide you with the classes to choose from.
        Just respond with the class you would assign to the text. Say nothing else. {exp_str}. \n An example is written below.
        """
        user_prompt = f"""text: {text}, classes: {classes}"""

        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                ]
        return messages
    
    def clf_batch_file(self, file_name:str, text_list:list, classes:list, give_explanation:bool=False, example:str=""):
        """
        Create a batch file for classification. 
        **NOTE: this is an asynchronous call, and it is recommended to use it for large requests."""

        save_path = os.path.join(self.script_path, 'batch requests', file_name)
        batch_request_path = os.path.join(self.script_path, 'batch requests')
        if file_name in os.listdir(batch_request_path):
            print('File already exists')
            raise FileExistsError(f'File with name *{file_name}* already exists')
        
        count = 0
        for text, cl in zip(text_list, classes):
            reqID = f"request-{count}"
            count += 1
            batch_format = {
            "custom_id":"",
            "method":"",
            "url":"",
            "body":{}
            }
            messages = self.clf_prompt(text, classes, give_explanation, example)
            batch_format["custom_id"] = reqID
            batch_format["method"] = "POST"
            batch_format["url"] = "/v1/chat/completions"
            batch_format["body"]["model"] = self.model_name
            batch_format["body"]["messages"] = messages
            batch_format["body"]["max_tokens"] = 2000

            
            if os.path.exists(save_path):
                print('File exists')
                with open(save_path, 'a') as file:
                    for entry in [batch_format]:
                        file.write(json.dumps(entry) + '\n')

            else:
                print('File does not exists')
                existing_data = {}
                for col in ['custom_id', 'method', 'url', 'body']:
                    existing_data[col] = batch_format[col]
                existing_data = [existing_data]
                with open(save_path, 'w') as file:
                    for entry in existing_data:
                        file.write(json.dumps(entry) + '\n')
        
    
    def clf_gen_batch(self, text_list:list, classes:list, give_explanation:bool=False, example:str="", description=""):
        """
        creating a batch of classifications.
        args:
        text: list, a list of texts to classify
        classes: list, a list of classes to choose from
        file_name: str, the name of the file containing the questions and answers
        description: str, the description of the batch request
        """
        file_name = str(uuid.uuid4()) + '.jsonl'
        self.clf_batch_file(file_name, text_list, classes, give_explanation, example)

        file_path = os.path.join(self.script_path, 'batch requests', file_name)
        batch_input_file = self.client.files.create(
        file=open(file_path, "rb"),
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

        task_specific_paramters = {'give_explanation':give_explanation, 'classes':classes}
        # save the metadata for the batch request
        log_batch_metadata(returned_json.id, batch_input_file_id, file_name, description,task='ANOT-CLASSIFICATION',task_specific_paramters=task_specific_paramters)
    

    # function to check if a batch is complete and matches the task to the response
    def check_batch_matches_task(self, batch_id:str, task:str):
        path = os.path.join(script_path, 'batch requests', 'logger.jsonl')
        all_meta_data = read_jsonl(path)
        batch_not_found = True
        job_task_dont_match = True
        for i in range(len(all_meta_data)):
            if all_meta_data[i]['id'] == batch_id:
                batch_not_found = False
                if task in all_meta_data[i]['task']:
                    job_task_dont_match = False
                break
        if batch_not_found:
            raise ValueError('Batch ID not found')
        if job_task_dont_match:
            raise ValueError(f'Batch ID {batch_id} does not match the task, task recorded as: {all_meta_data[i]["task"]}, target should be: ANOT-CLASSIFICATION')

    

    def create_clf_annotation_file(self, batch_id:str, file_path:str, overwrite:bool=False):
        """
        retireve the results of a batch request and save it as a jsonl file.
        **NOTE: file type accepted now is only .jsonl
        args:
        batch_id: str, the id of the batch request
        file_path: str, the path of the file to save the results
        """
        # update_batch_status(self.client)
        # status = get_batch_status(batch_id, self.client)
        # if status != 'completed':
        #     raise ValueError(f'Batch {batch_id} is not completed yet, current status is {status}')
        file_extension = file_path.split('.')[-1]
        if file_extension != 'jsonl':
            raise ValueError(f'File type is {file_extension}, only jsonl is supported at the moment.')
        # check if ID is for batch qa annotation
        path = os.path.join(script_path, 'batch requests', 'logger.jsonl')
        all_meta_data = read_jsonl(path)
        batch_not_found = True
        job_task_dont_match = True
        for i in range(len(all_meta_data)):
            if all_meta_data[i]['id'] == batch_id:
                batch_not_found = False
                if 'ANOT-CLASSIFICATION' in all_meta_data[i]['task']:
                    job_task_dont_match = False
                break
        if batch_not_found:
            raise ValueError('Batch ID not found')
        if job_task_dont_match:
            raise ValueError(f'Batch ID {batch_id} does not match the task, task recorded as: {all_meta_data[i]["task"]}, target should be: ANOT-CLASSIFICATION')
        matched_resp = match_response_to_request(batch_id, self.client)

        # writing to the path
        if os.path.exists(file_path):
            print(f'File {file_path} already exists')
            if overwrite:
                existing_data = {}
                for col in list(matched_resp.keys()):
                    existing_data[col] = matched_resp[col]
                existing_data = [existing_data]
                with open(file_path, 'w+') as file:
                    for entry in existing_data:
                        file.write(json.dumps(entry) + '\n')
            else:
                with open(file_path, 'a') as file:
                    for entry in [file_path]:
                        file.write(json.dumps(entry) + '\n')

        else:
            existing_data = {}
            for col in list(matched_resp.keys()):
                existing_data[col] = matched_resp[col]
            existing_data = [existing_data]
            with open(file_path, 'w+') as file:
                for entry in existing_data:
                    file.write(json.dumps(entry) + '\n')
    

    def parse_response_clf(self, batch_id:str=''):
        """
        Parse the response of a classification batch request.
        args:
        matched_response_file_path: str, the path to the file containing the matched response
        returns:
        parsed_response: dict, a dictionary containing the parsed response
        """
        self.check_batch_matches_task(batch_id, 'ANOT-CLASSIFICATION')
        matched_response = match_response_to_request(batch_id, self.client)
        task_specific_paramters = matched_response['task_specific_paramters']
        give_explanation = task_specific_paramters['give_explanation']
        classes = task_specific_paramters['classes']

        # parsing the response with re
        class_pattern = r"#class:\s*([^,]+)"
        rational_pattern = r"#rationale:\s*(.+)"


        parsed_response = {'input':[], 'class':[], 'rationale':[]}
        for i in range(len(matched_response['request'])):
            input_text = matched_response['request'][i]
            output = matched_response['response'][i]

            # Find matches
            if not give_explanation:
                class_match = re.search(class_pattern, output)
                rational_match = 'N/A'
                rational_value = 'N/A'
            else:
                class_match = re.search(class_pattern, output)
                rational_match = re.search(rational_pattern, output)
                class_value = class_match.group(1).strip()
                rational_value = rational_match.group(1).strip()
            
            # check if matches are found
            if class_value in classes:
                parsed_response['input'].append(input_text)
                parsed_response['class'].append(class_value)
                parsed_response['rationale'].append(rational_value)
            else:
                print(f'Class {class_value} not found in classes provided')


        return parsed_response


    # TODO: complete the function
    def function_calling_prompt(self, text:str, functions:dict={'function':[], 'parameters':[], 'description':[]}):
        system_prompt = f"""You are a helpful assistant calling a function. You are given a text and you need to call the appropriate function. The user will provide you with the functions to choose from.
        Just respond with the function you would call to the text. Say nothing else. An example is written below.
        User: get the weather in london for the next week.
        Function 1: get_weather, 
            parameters: 
                city: the city to get the weather
                time: the time to get the weather
            description: 
                get the weather for a city over the next week.
            
        Your response: function1
        """
        user_prompt = f"""text: {text}"""

        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                ]
        return messages

    # TODO: complete the function
    def function_calling_single(self, text:str, functions:dict={'function':[], 'description':[]}):
        pass
