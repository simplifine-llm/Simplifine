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
from logger import *
from numpy import array, zeros
import uuid
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from datasets import Dataset
from sentence_transformers.util import cos_sim
# import faiss
from utils import chunk_text_by_words



# TODO: attend to logging and make it streamlined.
class eval_gen:
    def __init__(self):

        # initilizing the local device
        self.device_name = None
        if torch.backends.mps.is_available():
            print('Using MPS')
            self.device = torch.device("mps")
            self.device_name = 'mps'
        elif torch.cuda.is_available():
            print('Using CUDA')
            self.device = torch.device("cuda")
            self.device_name = 'cuda'
        else:
            print('Using CPU')
            self.device = torch.device("cpu")
            self.device_name = 'cpu'


        self.client = OpenAI(
        api_key='',
        )
        self.model_name = 'gpt-4o'
        self.script_path = os.path.dirname(os.path.abspath(__file__))

        # check to see if file to store the batch requests exists
        if 'batch requests' not in os.listdir(self.script_path):
            print('Creating a directory to store batch requests')
            os.mkdir(os.path.join(self.script_path, 'batch requests'))
        
        self.summary_eval_prompt = """
                                You will be given one summary written for an article. Your task is to rate the summary on one metric.
                                Please make sure you read and understand these instructions very carefully. 
                                Please keep this document open while reviewing, and refer to it as needed.
                                Please only return the score and nothing else. This should be just the number, e.g. 3 and not - 3 or * 3.

                                Evaluation Criteria:

                                {criteria}

                                Evaluation Steps:

                                {steps}

                                Example:

                                Source Text:

                                {document}

                                Summary:

                                {summary}

                                Evaluation Form (scores ONLY):

                                - {metric_name}
                            """
        
    

    def create_prompt_single_qa(self, question:str, answer:str, give_explanation:bool, context:bool, example=None):
        system_prompt = f"""You are a helpful assistant assessing if an answer is correct or not.
        give a single word answer to the question. If the answer is correct, say "yes", if not, say "no".
        """
        user_prompt = f"""question: {question}, answer: {answer}"""

        if context:
            system_prompt += f"""\n the user also provides context about the question."""
            user_prompt += f""", context: {context}"""
        if give_explanation:
            system_prompt += f""" \ngive an explanation for you as answer. your answer should be formated as follows: #Answer: <answer>, #Explnation: <explnation>."""
        if example is not None:
            system_prompt += f"""\nHere is an example: {example}"""
        
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                ]
        return messages
            
    
    def QA_eval_single(self, question:str, answer:str, give_explanation:bool, context:bool, example=None):
        """
        evaluating a single question and answer pair.
        args:
        question: str, the question
        answer: str, the answer
        give_explanation: bool, whether to give an explanation or not
        example: str, an example of the question
        retruns:
        llm_answer: str, the answer from the model
        """
        
        messages = self.create_prompt_single_qa(question, answer, give_explanation, context, example)

        completion = self.client.chat.completions.create(
        model=self.model_name,
        messages=messages
        )
        llm_answer = completion.choices[0].message.content
        return llm_answer
    
    def create_batch_file_QA(self, questions:list, answers:list, give_explanation:bool, context:bool, example=None, file_name = 'batch_request_train.jsonl'):
        """
        creating a file containing a batch of questions and answers.
        args:
            questions: list, a list of questions
            answers: list, a list of answers
            give_explanation: bool, whether to give an explanation or not
            context: bool, whether to provide context or not
            example: str, an example of the question

        returns:
            file_name: str, the name of the file containing the questions and answers
        """

        # check if file exists
        save_path = os.path.join(self.script_path, 'batch requests', file_name)
        batch_request_path = os.path.join(self.script_path, 'batch requests')
        if file_name in os.listdir(batch_request_path):
            print('File already exists')
            raise FileExistsError(f'File with name *{file_name}* already exists')

        request_ids = [f"request-{i}" for i in range(len(questions))]
        for q,a,reqID in zip(questions, answers, request_ids):
            batch_format = {
            "custom_id":"",
            "method":"",
            "url":"",
            "body":{}
            }
            messages = self.create_prompt_single_qa(q, a, give_explanation, context, example)
            batch_format["custom_id"] = reqID
            batch_format["method"] = "POST"
            batch_format["url"] = "/v1/chat/completions"
            batch_format["body"]["model"] = self.model_name
            batch_format["body"]["messages"] = messages
            batch_format["body"]["max_tokens"] = 2000

            
            if os.path.exists(save_path):
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

    def QA_eval_batch(self, questions:list, answers:list, give_explanation:bool, context:bool, example=None, description=""):
        """
        evaluating a batch of questions and answers.
        args:
        file_name: str, the name of the file containing the questions and answers
        returns:
        llm_answers: list, a list of answers from the model
        """
        file_name = str(uuid.uuid4()) + '.jsonl'
        self.create_batch_file_QA(questions, answers, give_explanation, context, example, file_name)

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
        task_specific_paramters = {'give_explanation':give_explanation, 'context':context, 'example':example}
        # save the metadata for the batch request
        log_batch_metadata(returned_json.id, batch_input_file_id, file_name, description, task='EVAL-QA',task_specific_paramters=task_specific_paramters)
    

    def QA_eval_batch_get_metrics(self, batch_id:str):
        """
        getting the metrics for a batch request.
        args:
        batch_id: str, the id of the batch request
        returns:
        metrics: dict, a dictionary containing the metrics for the batch request
        """
        # check if the task is EVAL-QA
        task = get_task_from_batch_id(batch_id)
        if task != 'EVAL-QA':
            raise ValueError(f'The target task should be EVAL-QA, but it is recorded as {task}')
        status = self.client.batches.retrieve(batch_id).status
        if status != 'completed':
            raise ValueError('Batch request has not been completed yet')
        else:
            # format for matched_response: matched_response = {'custom_id':[], 'request':[], 'response':[]}
            matches_resp = match_response_to_request(batch_id, self.client)
            arr = zeros(len(matches_resp['response']))
            for i in range(len(matches_resp['response'])):
                resp = matches_resp['response'][i].lower()
                if 'yes' in resp:
                    arr[i] = 1
        return arr

    
    # TODO: extract the scores from the LLM response. 
    def summary_eval(self, input_text, generated_summary, metric_meta_data={'metric_name': [], 'metric':[], 'metric_scoring':[]}):
        """
        evaluating a summary of a text.
        **NOTE: This is a synchronus function and is not suitable for batch requests.
        args:
        input_text: str, the text to summarize
        generated_summary: summary to be evaluated
        metric_meta_data: dict, a dictionary containing the metric name, metric and metric scoring.
        returns:
        resp: The response from the LLM
        """
        
        scores = {}
        for i in range(len(metric_meta_data['metric_name'])):
            summary_eval_prompt = self.summary_eval_prompt.format(criteria=metric_meta_data['metric'][i], steps=metric_meta_data['metric_scoring'][i], document=input_text, summary=generated_summary, metric_name=metric_meta_data['metric_name'][i])
            messages = [
                {"role": "system", "content": summary_eval_prompt},
                {"role": "user", "content": input_text}
            ]
            completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
            )
            resp = completion.choices[0].message.content
            print(f'{metric_meta_data["metric_name"][i]}: {resp}')
            scores[metric_meta_data['metric_name'][i]] = int(resp)
        return scores


    def summary_eval_batch(self, file_name:str, input_texts:list, generate_summaries:list, metric_meta_data={'metric_name': [], 'metric':[], 'metric_scoring':[]}, description:str=''):
        """
        evaluating a batch of summaries.
        args:
        input_texts: list, a list of texts to summarize
        generate_summaries: list, a list of summaries to evaluate
        metric_meta_data: dict, a dictionary containing the metric name, metric and metric scoring.
        returns:
        resp: The response from the LLM
        """
        save_path = os.path.join(self.script_path, 'batch requests', file_name)
        batch_request_path = os.path.join(self.script_path, 'batch requests')
        if file_name in os.listdir(batch_request_path):
            print('File already exists')
            raise FileExistsError(f'File with name *{file_name}* already exists')
        
        for text, summ, ind in zip(input_texts, generate_summaries, range(len(input_texts))):
            cur_req_id = f"request-{ind}"
            for i in range(len(metric_meta_data['metric_name'])):
                summary_eval_prompt = self.summary_eval_prompt.format(criteria=metric_meta_data['metric'][i], steps=metric_meta_data['metric_scoring'][i], document=text, summary=summ, metric_name=metric_meta_data['metric_name'][i])
                messages = [
                    {"role": "system", "content": summary_eval_prompt},
                    {"role": "user", "content": text}
                ]
                batch_format = {
                "custom_id":"",
                "method":"",
                "url":"",
                "body":{}
                }
                batch_format["custom_id"] = cur_req_id+f'-{i}'
                batch_format["method"] = "POST"
                batch_format["url"] = "/v1/chat/completions"
                batch_format["body"]["model"] = self.model_name
                batch_format["body"]["messages"] = messages
                batch_format["body"]["max_tokens"] = 2000

                
                if os.path.exists(save_path):
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
        

        batch_input_file = self.client.files.create(
        file=open(save_path, "rb"),
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
        task_specific_paramters = {'metric_meta_data':metric_meta_data}
        # save the metadata for the batch request
        log_batch_metadata(returned_json.id, batch_input_file_id, file_name, description, task='EVAL-SUMMARIZATION', task_specific_paramters=task_specific_paramters)
    

    def clf_eval(self, y_pred:list, y_label:list):
        """
        evaluating a classification model.
        args:
        y_pred: list, the predicted labels
        y_label: list, the true labels
        returns:
        arr: an array denoting 1 where the prediction is correct and 0 where it is not.
        """
        inds = [num for num in range(len(y_pred)) if y_pred[num] == y_label[num]]
        arr = zeros(len(y_pred))
        arr[inds] = 1
        return arr
    

    def eval_embedder_with_pairs(self, queries:list=[], 
                      context:list=[],
                      embedder:str='paraphrase-MiniLM-L6-v2',
                      relevant_ids:list=[],):
        
        """
        This function evalutes an embedder using the information retrieval evaluator.
        args:
        queries: list, a list of queries
        context: list, a list of context
        embedder: str, the embedder to evaluate
        returns:
        results: dict, the results of the evaluation
        """
        # TODO:insert this vars into the func args
        matryoshka_dimensions = [128, 256]
        use_matryoshka = False


        # initialize the embedder
        # check if the model is available on the GPU, then implemenbt flash attention 2
        if self.device_name == 'cuda':
            model = SentenceTransformer(
            embedder, 
            model_kwargs={"attn_implementation": "sdpa"}
            ,trust_remote_code=True
            )
        else:
            model = SentenceTransformer(
            embedder, device=self.device,
            trust_remote_code=True
            )

        
        dataset_q = Dataset.from_dict({'queries':queries})
        dataset_c = Dataset.from_dict({'docs':context})
        dataset_q = dataset_q.add_column("id", range(len(dataset_q)))
        dataset_c = dataset_c.add_column("id", range(len(dataset_c)))

    
        # Convert the datasets to dictionaries
        corpus = dict(
            zip(dataset_c["id"], dataset_c["docs"])
        )  # Our corpus (cid => document)
        queries = dict(
            zip(dataset_q["id"], dataset_q["queries"])
        )  # Our queries (qid => question

        relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
        for num,q_id in enumerate(queries):
            relevant_docs[q_id] = [relevant_ids[num]]
        
        print('f------------------- relevant docs in the function: ', relevant_docs)

        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="information-retrieval-evaluator",
            score_functions={"cosine": cos_sim},
        )

        if use_matryoshka:
            matryoshka_evaluators = []
            # Iterate over the different dimensions
            for dim in matryoshka_dimensions:
                ir_evaluator = InformationRetrievalEvaluator(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=relevant_docs,
                    name=f"dim_{dim}",
                    truncate_dim=dim,  # Truncate the embeddings to a certain dimension
                    score_functions={"cosine": cos_sim},
                )
                matryoshka_evaluators.append(ir_evaluator)
            evaluator = SequentialEvaluator(matryoshka_evaluators)

        results = evaluator(model)
        return results
    

    # ==============================================================================================================
    # eval_embedder_openAI_prompt_creation but per context
    # TODO: faiss not working, fix it!!!
    def eval_embedder_openAI_prompt_creation_per_ctx(self, text:str='',
                                queries:list=[],
                                embedder:str='paraphrase-MiniLM-L6-v2',
                                chunk_size:int=4,
                                overlap:int=2,
                                top_k:int=5,
                                verbose:bool=True,
                                use_faiss:bool=False,
                                relevant_chunks_provided:bool=False,
                                chunks:list=[]):
        
        system_prompt = f"""You are a helpful assistant assessing if the context provided by the user is relevant to the query.
        The user provides a query and a context. The user asks you to determine if the context is relevant to the query.
        Relevant context is one that answers the query. Irrelevant context is one that does not answer the query.
        The user provides several contexts. If the write answer can be found in one of the contexts, the context is relevant and you should say "yes".
        If the correct answer cannot be found in any of the contexts, the context is irrelevant and you should say "no".
        an example is written below:
        User: Quey: what is the capitla of France? 
        Context: Paris has been the capital of France since the 5th century.
        Your response: Yes

        Another example:
        Query: What is the name of the company CEO?
        Context: X corp is a company that was founded in 1999. 
        Your response: No
        """

        
        if relevant_chunks_provided:
            # generating the text
            strings_to_validation = []
            for num,q in enumerate(queries):
                relevant_chunks = chunks[num]
                cur_strings = []
                for i in range(len(relevant_chunks)):
                    string = f'Query: {q}\n'
                    string += f'Context: {relevant_chunks[i]}\n'
                    cur_strings.append(string)
                strings_to_validation.append(cur_strings)
        else:
            # initialize the embedder
            # check if the model is available on the GPU, then implemenbt flash attention 2
            if self.device_name == 'cuda':
                model = SentenceTransformer(
                embedder, 
                model_kwargs={"attn_implementation": "sdpa"}
                ,trust_remote_code=True
                )
            else:
                model = SentenceTransformer(
                embedder, device=self.device,
                trust_remote_code=True
                )

            # returns a numpy array of the embeddings
            if verbose:
                print('Embedding the text...')
            embeddings_queries = model.encode(queries)

            chunks = chunk_text_by_words(text, chunk_size, overlap)
            embeddings_chunks = model.encode(chunks)
            if use_faiss:
                print('FAISS not availble in this version.')
                # if verbose:
                #     print('indexing using FAISS...')
                # index = faiss.IndexFlatL2(embeddings_chunks.shape[-1])   # build the index
                # index.add(embeddings_chunks)                  # add vectors to the index
                # if verbose:
                #     print('searching for the most similar chunks...')
                # D, I = index.search(embeddings_queries, top_k) # actual search
                # results_index = I
            
            else:
                if verbose:
                    print('searching for the most similar chunks...')
                results = cos_sim(embeddings_queries, embeddings_chunks)
                results_index = torch.topk(results, top_k, dim=1).indices.numpy()
            
            # generating the text
            strings_to_validation = []
            for num,q in enumerate(queries):
                relevant_chunks = chunks[num]
                cur_strings = []
                for i in range(len(relevant_chunks)):
                    string = f'Query: {q}\n'
                    string += f'Context: {relevant_chunks[i]}\n'
                    cur_strings.append(string)
                strings_to_validation.append(cur_strings)
        
        user_prompts = []
        for string_per_ctx in strings_to_validation:
            cur_user_prompts = []
            for string in string_per_ctx:
                user_prompt = f"""{string}"""
                cur_user_prompts.append(user_prompt)
            user_prompts.append(cur_user_prompts)

        return user_prompts, system_prompt
    
    def sync_embed_eval_per_ctx(self, text:str='',
                                queries:list=[],
                                embedder:str='paraphrase-MiniLM-L6-v2',
                                chunk_size:int=4,
                                overlap:int=2,
                                top_k:int=5,
                                verbose:bool=True,
                                use_faiss:bool=False,
                                relevant_chunks_provided:bool=True,
                                chunks:list=[]):
        """
        sync calls to openAI for evaluation of an embedder.
        NOTE: This function is synchronous and is suitable for small datasets and quick requests.
        args:
        text: str, the text to embed
        queries: list, a list of queries
        embedder: str, the embedder to evaluate
        chunk_size: int, the size of the chunks
        overlap: int, the overlap between the chunks
        top_k: int, the number of top k results to return
        verbose: bool, whether to print the progress of the function
        use_faiss: bool, whether to use faiss for the search
        returns:
        matched_response: dict, the matched response from the model
        """
        user_prompts, system_prompt = self.eval_embedder_openAI_prompt_creation_per_ctx(text, queries, embedder, chunk_size, overlap, top_k, verbose, use_faiss, relevant_chunks_provided, chunks)
        responses = []
        for num, user_prompt in enumerate(user_prompts):
            cur_response = []
            for string in user_prompt:
                messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": string}
                ]
                completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
                )
                llm_answer = completion.choices[0].message.content
                cur_response.append(llm_answer)
            responses.append(cur_response)
        return responses
    

    def async_embed_eval_per_ctx(self, text:str='',
                                queries:list=[],
                                embedder:str='paraphrase-MiniLM-L6-v2',
                                chunk_size:int=4,
                                overlap:int=2,
                                top_k:int=5,
                                verbose:bool=True,
                                use_faiss:bool=False,
                                relevant_chunks_provided:bool=True,
                                chunks:list=[]):
        """
        creating a batch request to evaluate an embedder.
        NOTE: This function is asynchronous and is suitable for batch requests and large datasets.
        args:
        text: str, the text to embed
        queries: list, a list of queries
        embedder: str, the embedder to evaluate
        chunk_size: int, the size of the chunks
        overlap: int, the overlap between the chunks
        top_k: int, the number of top k results to return
        verbose: bool, whether to print the progress of the function
        use_faiss: bool, whether to use faiss for the search
        returns:
        matched_response: dict, the matched response from the model
        """
        user_prompts, system_prompt = self.eval_embedder_openAI_prompt_creation_per_ctx(text, queries, embedder, chunk_size, overlap, top_k, verbose, use_faiss, relevant_chunks_provided, chunks)
        file_name = str(uuid.uuid4()) + '.jsonl'
        save_path = os.path.join(self.script_path, 'batch requests', file_name)
        batch_request_path = os.path.join(self.script_path, 'batch requests')
        if file_name in os.listdir(batch_request_path):
            print('File already exists')
            raise FileExistsError(f'File with name *{file_name}* already exists')
        
        for num, user_prompt in enumerate(user_prompts):
            for num2, string in enumerate(user_prompt):
                messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": string}
                ]
                batch_format = {
                "custom_id":"",
                "method":"",
                "url":"",
                "body":{}
                }
                batch_format["custom_id"] = f"request-{num}{num2}"
                batch_format["method"] = "POST"
                batch_format["url"] = "/v1/chat/completions"
                batch_format["body"]["model"] = self.model_name
                batch_format["body"]["messages"] = messages
                batch_format["body"]["max_tokens"] = 100
        
                if os.path.exists(save_path):
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

        batch_input_file = self.client.files.create(
        file=open(save_path, "rb"),
        purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        returned_json = self.client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description":"Evaluation of an embedder per context provided"
        }
        )
        task_specific_paramters = {'embedder':embedder, 'chunk_size':chunk_size, 'overlap':overlap, 'top_k':top_k, 'use_faiss':use_faiss}
        # save the metadata for the batch request
        log_batch_metadata(returned_json.id, batch_input_file_id, file_name, "Evaluation of an embedder", task='EVAL-EMBEDDER', task_specific_paramters=task_specific_paramters)
    
    
    # ==============================================================================================================


    # TODO: faiss not working, fix it!!!
    def eval_embedder_openAI_prompt_creation(self, text:str='',
                                queries:list=[],
                                embedder:str='paraphrase-MiniLM-L6-v2',
                                chunk_size:int=4,
                                overlap:int=2,
                                top_k:int=5,
                                verbose:bool=True,
                                use_faiss:bool=False,
                                relevant_chunks_provided:bool=True,
                                chunks:list=[]):
        
        if relevant_chunks_provided:
            # generating the text
            strings_to_validation = []
            for num,q in enumerate(queries):
                string = f'Query: {q}\n'
                relevant_chunks = chunks[num]
                for i in range(len(relevant_chunks)):
                    string += f'Context {i+1}: {relevant_chunks[i]}\n'
                strings_to_validation.append(string)
        else:
            # initialize the embedder
            # check if the model is available on the GPU, then implemenbt flash attention 2
            if self.device_name == 'cuda':
                model = SentenceTransformer(
                embedder, 
                model_kwargs={"attn_implementation": "sdpa"}
                ,trust_remote_code=True
                )
            else:
                model = SentenceTransformer(
                embedder, device=self.device,
                trust_remote_code=True
                )

            # returns a numpy array of the embeddings
            if verbose:
                print('Embedding the text...')
            embeddings_queries = model.encode(queries)

            chunks = chunk_text_by_words(text, chunk_size, overlap)
            embeddings_chunks = model.encode(chunks)
            if use_faiss:
                print("FAISS not available in this version.")
                # if verbose:
                #     print('indexing using FAISS...')
                # index = faiss.IndexFlatL2(embeddings_chunks.shape[-1])   # build the index
                # index.add(embeddings_chunks)                  # add vectors to the index
                # if verbose:
                #     print('searching for the most similar chunks...')
                # D, I = index.search(embeddings_queries, top_k) # actual search
                # results_index = I
            
            else:
                if verbose:
                    print('searching for the most similar chunks...')
                results = cos_sim(embeddings_queries, embeddings_chunks)
                results_index = torch.topk(results, top_k, dim=1).indices.numpy()
            
            # generating the text
            strings_to_validation = []
            for num,q in enumerate(queries):
                string = f'Query: {q}\n'
                relevant_chunks = [chunks[i] for i in results_index[num]]
                for i in range(len(relevant_chunks)):
                    string += f'Context {i+1}: {relevant_chunks[i]}\n'
                strings_to_validation.append(string)
        
        system_prompt = f"""You are a helpful assistant assessing if the context provided by the user is relevant to the query.
        The user provides a query and a context. The user asks you to determine if the context is relevant to the query.
        Relevant context is one that answers the query. Irrelevant context is one that does not answer the query.
        The user provides several contexts. If the write answer can be found in one of the contexts, the context is relevant and you should say "yes".
        If the correct answer cannot be found in any of the contexts, the context is irrelevant and you should say "no".
        an example is written below:
        User: Quey: what is the capitla of France? 
        Context 1: Paris has been the capital of France since the 5th century.
        Context 2: The Eiffel Tower is located in Paris.
        Context 3: The Louvre Museum is located in Paris.
        Context 4: The Seine River runs through Paris.
        Your response: Yes
        """
        user_prompts = []
        for string in strings_to_validation:
            user_prompt = f"""{string}"""
            user_prompts.append(user_prompt)

        return user_prompts, system_prompt
    
    def sync_embed_eval(self, text:str='',
                                queries:list=[],
                                embedder:str='paraphrase-MiniLM-L6-v2',
                                chunk_size:int=4,
                                overlap:int=2,
                                top_k:int=5,
                                verbose:bool=True,
                                use_faiss:bool=False,
                                relevant_chunks_provided:bool=True,
                                chunks:list=[]):
        """
        sync calls to openAI for evaluation of an embedder.
        NOTE: This function is synchronous and is suitable for small datasets and quick requests.
        args:
        text: str, the text to embed
        queries: list, a list of queries
        embedder: str, the embedder to evaluate
        chunk_size: int, the size of the chunks
        overlap: int, the overlap between the chunks
        top_k: int, the number of top k results to return
        verbose: bool, whether to print the progress of the function
        use_faiss: bool, whether to use faiss for the search
        returns:
        matched_response: dict, the matched response from the model
        """
        user_prompts, system_prompt = self.eval_embedder_openAI_prompt_creation(text, queries, embedder, chunk_size, overlap, top_k, verbose, use_faiss, relevant_chunks_provided, chunks)
        responses = []
        for num, user_prompt in enumerate(user_prompts):
            messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ]
            completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
            )
            llm_answer = completion.choices[0].message.content
            responses.append(llm_answer)
        return responses
    
    def async_embed_eval(self, text:str='',
                                queries:list=[],
                                embedder:str='paraphrase-MiniLM-L6-v2',
                                chunk_size:int=4,
                                overlap:int=2,
                                top_k:int=5,
                                verbose:bool=True,
                                use_faiss:bool=False,
                                relevant_chunks_provided:bool=True,
                                chunks:list=[]):
        """
        creating a batch request to evaluate an embedder.
        NOTE: This function is asynchronous and is suitable for batch requests and large datasets.
        args:
        text: str, the text to embed
        queries: list, a list of queries
        embedder: str, the embedder to evaluate
        chunk_size: int, the size of the chunks
        overlap: int, the overlap between the chunks
        top_k: int, the number of top k results to return
        verbose: bool, whether to print the progress of the function
        use_faiss: bool, whether to use faiss for the search
        returns:
        matched_response: dict, the matched response from the model
        """
        user_prompts, system_prompt = self.eval_embedder_openAI_prompt_creation(text, queries, embedder, chunk_size, overlap, top_k, verbose, use_faiss, relevant_chunks_provided, chunks)
        file_name = str(uuid.uuid4()) + '.jsonl'
        save_path = os.path.join(self.script_path, 'batch requests', file_name)
        batch_request_path = os.path.join(self.script_path, 'batch requests')
        if file_name in os.listdir(batch_request_path):
            print('File already exists')
            raise FileExistsError(f'File with name *{file_name}* already exists')
        
        for num, user_prompt in enumerate(user_prompts):
            messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ]
            batch_format = {
            "custom_id":"",
            "method":"",
            "url":"",
            "body":{}
            }
            batch_format["custom_id"] = f"request-{num}"
            batch_format["method"] = "POST"
            batch_format["url"] = "/v1/chat/completions"
            batch_format["body"]["model"] = self.model_name
            batch_format["body"]["messages"] = messages
            batch_format["body"]["max_tokens"] = 100
    
            if os.path.exists(save_path):
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

        batch_input_file = self.client.files.create(
        file=open(save_path, "rb"),
        purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        returned_json = self.client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description":"Evaluation of an embedder"
        }
        )
        task_specific_paramters = {'text':text, 'queries':queries, 'embedder':embedder, 'chunk_size':chunk_size, 'overlap':overlap, 'top_k':top_k, 'verbose':verbose, 'use_faiss':use_faiss}
        # save the metadata for the batch request
        log_batch_metadata(returned_json.id, batch_input_file_id, file_name, "Evaluation of an embedder", task='EVAL-EMBEDDER', task_specific_paramters=task_specific_paramters)
    

    def parse_async_eval_embed_openAI(self, batch_id:str=''):
        if batch_id == '':
            raise ValueError('Please provide a valid batch id')
        task = get_task_from_batch_id(batch_id)
        if task != "EVAL-EMBEDDER":
            raise ValueError(f'Task is not EVAL-EMBEDDER, instead it is {task}')
        
        status = self.client.batches.retrieve(batch_id).status
        if status != 'completed':
            ValueError(f'batch not complete, status is {status}')

        matches_resp = match_response_to_request(batch_id, self.client)
        return matches_resp
    

    def eval_hallucination_rag_prompt(self,
                               query:str,
                               contexts:str,
                               answer:str,
                               num_outputs:int,
                               based_on_context:bool,):
        """
        evaluting hallucination in a RAG pipe.
        """
        system_prompt = f"""You are a helpful assistant assessing if an answer given to a question based on context includes hallucinations or not.
        The user's input has the following format: Query: <query>\n Context 1: <context>\n Context 2: <context>\n Context 3: <context>\n Context 4: <context>\n Context 5: <context> \n Answer: <answer>.
        The user can provide different number of contexts. Your task is to determine if the answer to the query based on the context includes hallucinations or not.
        Respond with "yes" if the answer includes hallucinations and "no" if the answer does not include hallucinations.
        """
        user_prompt = f'Query: {query}\n'
        for num,context in enumerate(contexts):
            user_prompt += f'Context {num}: {context}\n'
        user_prompt += f'Answer: {answer}'
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages
    
    def eval_hallucination_rag_sync(self,
                               query:str,
                               contexts:str,
                               answer:str,
                               num_outputs:int,
                               based_on_context:bool):
        """
        evaluating hallucination in a RAG pipe synchronously.
        NOTE: This function is synchronous and is suitable for signle data points.
        """
        messages = self.eval_hallucination_rag_prompt(query, contexts, answer, num_outputs, based_on_context)
        completion = self.client.chat.completions.create(
        model=self.model_name,
        messages=messages
        )
        llm_answer = completion.choices[0].message.content
        return llm_answer
    
    def eval_hallucination_rag_async(self,
                               queries:list,
                               contexts:list,
                               answers:list,
                               num_outputs:int,
                               based_on_context:bool):
        """
        evaluating hallucination in a RAG pipe asynchronously.
        NOTE: This function is asynchronous and is suitable for batch requests and large datasets.
        """
        file_name = str(uuid.uuid4()) + '.jsonl'
        save_path = os.path.join(self.script_path, 'batch requests', file_name)
        batch_request_path = os.path.join(self.script_path, 'batch requests')
        if file_name in os.listdir(batch_request_path):
            print('File already exists')
            raise FileExistsError(f'File with name *{file_name}* already exists')
        
        for num, (query, context, answer) in enumerate(zip(queries, contexts, answers)):
            messages = self.eval_hallucination_rag_prompt(query, context, answer, num_outputs, based_on_context)
            batch_format = {
            "custom_id":"",
            "method":"",
            "url":"",
            "body":{}
            }
            batch_format["custom_id"] = f"request-{num}"
            batch_format["method"] = "POST"
            batch_format["url"] = "/v1/chat/completions"
            batch_format["body"]["model"] = self.model_name
            batch_format["body"]["messages"] = messages
            batch_format["body"]["max_tokens"] = 100
    
            if os.path.exists(save_path):
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
        
        batch_input_file = self.client.files.create(
        file=open(save_path, "rb"),
        purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        returned_json = self.client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description":"Evaluation of hallucination in a RAG pipe"
        }
        )
        task_specific_paramters = {}
        # save the metadata for the batch request
        log_batch_metadata(returned_json.id, batch_input_file_id, file_name, "Evaluation of hallucination in a RAG pipe", task='EVAL-HALLUCINATION-RAG', task_specific_paramters=task_specific_paramters)
    
    def parse_async_eval_hallucination_rag(self, batch_id:str=''):
        if batch_id == '':
            raise ValueError('Please provide a valid batch id')
        task = get_task_from_batch_id(batch_id)
        if task != "EVAL-HALLUCINATION-RAG":
            raise ValueError(f'Task is not EVAL-HALLUCINATION-RAG, instead it is {task}')
        
        status = self.client.batches.retrieve(batch_id).status
        if status != 'completed':
            ValueError(f'batch not complete, status is {status}')

        matches_resp = match_response_to_request(batch_id, self.client)
        return matches_resp
