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
import json
import os
from openai import OpenAI
from utils import read_jsonl
import ast

script_path = os.path.dirname(os.path.abspath(__file__))


def get_task_from_batch_id(batch_id:str):
    """
    getting the task from the batch id.
    args:
    batch_id: str, the id of the batch request
    returns:
    task: str, the task of the batch request
    """
    filename = os.path.join(script_path, 'batch requests', "logger.jsonl")
    all_meta = read_jsonl(filename)
    for i in range(len(all_meta)):
        if all_meta[i]['id'] == batch_id:
            task = all_meta[i]['task']
            return task
    raise Exception("The batch request id is not valid or the file is not found.")


def write_jsonl(file_path, data, append=True):
    """
    Writes a list of JSON serializable objects to a JSONL file.
    
    :param file_path: The path to the JSONL file.
    :param data: The list of JSON serializable objects.
    :param append: Whether to append to the file or overwrite it.
    """
    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            for entry in [file_path]:
                file.write(json.dumps(entry) + '\n')

    else:
        existing_data = {}
        for col in list(data.keys()):
            existing_data[col] = data[col]
        existing_data = [existing_data]
        with open(file_path, 'w+') as file:
            for entry in existing_data:
                file.write(json.dumps(entry) + '\n')

    
def log_batch_metadata(batch_id, input_file_id, input_file_name, description, task, task_specific_paramters=[]):
    """
    logging the metadata for a batch request.
    args:
    batch_id: str, the id of the batch request
    input_file_id: str, the id of the input file
    input_file_name: str, the name of the input file
    description: str, the description of the batch request
    """
    metadata = {
    "id": batch_id, 
    "input_file_id": input_file_id,
    "input_file_name": input_file_name,
    "description": description,
    "status": "NA",
    'task': task,
    'task_specific_paramters': task_specific_paramters,
    }

    filename = os.path.join(script_path, 'batch requests', "logger.jsonl")

    if os.path.exists(filename):
        with open(filename, 'a') as file:
            for entry in [metadata]:
                file.write(json.dumps(entry) + '\n')

    else:
        existing_data = {}
        for col in metadata.keys():
            existing_data[col] = metadata[col]
        existing_data = [existing_data]
        with open(filename, 'w+') as file:
            for entry in existing_data:
                file.write(json.dumps(entry) + '\n')
    



def get_batch_status(client, batch_id):
    """
    getting the status of a batch request.
    args:
    batch_id: str, the id of the batch request
    returns:
    status: str, the status of the batch request
    """
    status = client.batches.retrieve(batch_id).status
    return status

def get_batch_meta(client, batch_id):
    """
    getting the status of a batch request.
    args:
    batch_id: str, the id of the batch request
    returns:
    status: str, the status of the batch request
    """
    batch_meta_data = client.batches.retrieve(batch_id)
    return batch_meta_data

def retireve_batch_results(client, batch_id):
    """
    retrieving the results of a batch request.
    args:
    batch_id: str, the id of the batch request
    returns:
    results: list, a list of results from the batch request
    """
    status_meta_data = get_batch_meta(client, batch_id)
    if status_meta_data.status == "completed":
        content = client.files.content(status_meta_data.output_file_id).content.decode('utf-8')
        return content
    else:
        raise Exception("The batch request is not completed yet.")
    

def retireve_batch_input(batch_id:str, return_task_parameters:bool=True):
    """
    retrieving the input of a batch request.
    args:
    batch_id: str, the id of the batch request
    returns:
    results: list, a list of results from the batch request
    """
    filename = os.path.join(script_path, 'batch requests', "logger.jsonl")
    all_meta = read_jsonl(filename)
    input_file_name = None
    for i in range(len(all_meta)):
        if all_meta[i]['id'] == batch_id:
            input_file_name = all_meta[i]['input_file_name']
            task_params = all_meta[i]['task_specific_paramters']
    if input_file_name is None:
        raise Exception("The batch request id is not valid or the file is not found.")
    input_file_path = os.path.join(script_path, 'batch requests', input_file_name)
    meta_data = read_jsonl(input_file_path)
    if return_task_parameters:
        return meta_data, task_params
    else:
        return meta_data
    

def update_batch_status_signle(client, batch_id):
    """
    function to update the status of a single batch request in the logger.jsonl file.
    args:
    client: OpenAI, the OpenAI client
    batch_id: str, the id of the batch request
    """
    path = os.path.join(script_path, 'batch requests', 'logger.jsonl')
    all_meta_data = read_jsonl(path)
    for i in range(len(all_meta_data)):
        if all_meta_data[i]['id'] == batch_id:
            all_meta_data[i]['status'] = get_batch_status(client, batch_id)
    
    with open(path, 'w') as file:
        for entry in all_meta_data:
            file.write(json.dumps(entry)) + '\n'




def update_batch_status(client):
    """
    function to update the status of all batch requests in the logger.jsonl file.
    args:
    client: OpenAI, the OpenAI client
    """
    path = os.path.join(script_path, 'batch requests', 'logger.jsonl')
    all_meta_data = read_jsonl(path)
    for i in range(len(all_meta_data)):
        if all_meta_data[i]['status'] != 'completed' or all_meta_data[i]['status'] != 'failed' or all_meta_data[i]['status'] != 'canceled' or all_meta_data[i]['status'] != 'expired':
            all_meta_data[i]['status'] = get_batch_status(client, all_meta_data[i]['id'])
    
    with open(path, 'w') as file:
        for entry in all_meta_data:
            file.write(json.dumps(entry) + '\n')

def match_response_to_request(batch_id, client):
    """
    match the response to the request.
    args:
        response: dict, the response from the model
        request: dict, the request to the model
    returns:
        matched_response: dict, the matched response
    """
    # converting the output from openAI to json/list
    cont = retireve_batch_results(client, batch_id)
    resps = []
    for i in cont.split('\n'):
        if len(i)>5:
            resps.append(json.loads(i))
    
    # getting the input data
    reqs, task_specific_paramters = retireve_batch_input(batch_id)

    matched_response = {'custom_id':[], 'request':[], 'response':[], 'task_specific_paramters':task_specific_paramters}
    # TODO: optimize this
    for resp in resps:
        cur_id = resp['custom_id']
        matched_response['custom_id'].append(cur_id)
        for req in reqs:
            if req['custom_id'] == cur_id:
                cur_resp = resp['response']['body']['choices'][0]['message']['content']
                cur_req = req['body']['messages'][-1]['content']
                matched_response['request'].append(cur_req)
                matched_response['response'].append(cur_resp)
                break
    return matched_response
    

if __name__ == '__main__':
    client = OpenAI(
        api_key='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',
    )
    batch_id = 'batch_hGsLl1EwGbyPO3PatxFX19Mj'
    print(match_response_to_request(batch_id, client))

