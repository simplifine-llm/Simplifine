from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import numpy as np
from utils import chunk_text_by_words, read_pdf
from sentence_transformers.util import cos_sim
import torch
from annotaion import synthetic
from sklearn.utils import shuffle
from eval import eval_gen
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from pinecone import Pinecone, ServerlessSpec
import os
import time
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import json
from openai import OpenAI




class pincecone_db_man:
    def __init__(self,
                index_name:str='tt4',
                namespace:str='tt5',
                model_name:str='text-embedding-3-small',
                openai_api_key:str=''
                ):
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.model_name = model_name
        self.open_ai_api_key = openai_api_key
        pinecone_api = '2c2648f9-7438-4299-a868-aa1cb3dcd07f'
        os.environ['PINECONE_API_KEY'] = pinecone_api
        self.pc = Pinecone(pinecone_api)
        self.index_name = index_name
        self.namespace = namespace
        self.index = self.create_index()
        self.embedder = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=openai_api_key
        )
        self.log_data = {'id':[], 'content':[]}
        self.log_file = 'log.json'
        self.log_file_path = os.path.join(self.script_path, self.log_file)
        if os.path.exists(self.log_file_path):
            # File exists, read the existing content
            with open(self.log_file_path, 'r') as file:
                self.log_data = json.load(file)
        
        self.openai_client = OpenAI(
        api_key='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',
        )
        


    def log(self, data):
        if os.path.exists(self.log_file_path):
            # File exists, read the existing content
            with open(self.log_file_path, 'r') as file:
                existing_data = json.load(file)
            
            # Combine the existing data with the new data
            combined_data = {
                'id': existing_data['id'] + data['id'],
                'content': existing_data['content'] + data['content']
            }
        else:
            # File does not exist, use the provided data
            combined_data = data

        # Save the combined data back to the file
        with open(self.log_file_path, 'w') as file:
            json.dump(combined_data, file, indent=4)
        
        # update the log data
        self.log_data = combined_data

    def create_index(self, dimension:int=1536, metric:str='cosine'):
        if  self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name= self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
            )
            # wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
        
        return self.pc.Index(self.index_name)
            
    
    def embed_data(self, data:list):
        return self.embedder.embed_documents(data)

    def embed_query(self, query:str):
        return self.embedder.embed_query(query)

    def add_data(self, data:list, log_local:bool=False):
        data_embed = self.embed_data(data)
        zero_ind = self.index.describe_index_stats()['total_vector_count']
        zero_ind+=1
        vectors = []
        log_data = {'id':[], 'content':[]}
        for i, d in enumerate(data):
            vectors.append({'id':str(i+zero_ind), 'values':data_embed[i]})
            log_data['id'].append(str(i+zero_ind))
            log_data['content'].append(d)
        self.index.upsert(
            vectors=vectors,
            namespace=self.namespace
        )
        if log_local:
            self.log(log_data)
        

    
    def search(self, query:str, topk:int=5):
        query_embed = self.embed_query(query)
        query_results = self.index.query(
        namespace=self.namespace,
        vector=query_embed,
        top_k=topk,
        include_values=True
        )
        return query_results
    
    def ask(self, question:str='', topk:int=5):
        search_res = self.search(question, topk)
        inds = []
        for i in search_res['matches']:
            inds.append(i['id'])
        
        # find the content of the matches
        content = []
        for i in inds:
            content.append(self.log_data['content'][int(i)-1])
        
        return self.invoke_openai(question, content)

    def invoke_openai(self, question:str, context:list):

        messages = [
        {
            "role": "system",
            "content": f'''The user has provided the following context and will be asking a qeustion about it: {' '.join(context)}'''
        },
        {
            "role": "user",
            "content": question
        }
        ]

        completion = self.openai_client.chat.completions.create(
        model='gpt-4o',
        messages=messages
        )

        llm_answer = completion.choices[0].message.content
        return llm_answer
        
    


class piencone_db:
    def __init__(self, 
                 index_name:str='a',
                 namespace:str='b',
                 path:str='',
                 text:str='',
                 mode:str='text'
                 ) -> None:
        pinecone_api = ''
        openai_api =''
        model_name = 'text-embedding-3-small'
        self.path = path

        os.environ['PINECONE_API_KEY'] = pinecone_api

        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=openai_api
        )

        self.pc = Pinecone(pinecone_api)
        spec = ServerlessSpec(cloud='aws', region='us-east-1')
        self.index_name = index_name
        self.namespace = namespace

        if  self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name= self.index_name,
                dimension=1536,
                metric="cosine",
                spec=spec
            )
            # wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
        self.llm = ChatOpenAI(
        openai_api_key=openai_api,
        model_name='gpt-3.5-turbo',
        temperature=0.0
        )
        docsearch = self.init_vectorstore(path=self.path, mode=mode, text=text)
        self.qa = RetrievalQA.from_chain_type(
        llm=self.llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever()
        )
    
    def init_vectorstore(self, mode:str='pdf', path:str='', text:str='', chunk_size:int=100, overlap:int=10):

        # if mode == 'web':
        #     pass
        
        # if mode == 'pdf':
        #     text = read_pdf(path)
        
        # text_splitter = CharacterTextSplitter(
        #     separator = ".",
        #     chunk_size = chunk_size,
        #     chunk_overlap  = overlap
        #     )
        
        # docs = text_splitter.create_documents([text])
        docsearch = PineconeVectorStore(
            index_name= self.index_name,
            embedding=self.embeddings,
            namespace=self.namespace
        )
        time.sleep(1)
        return docsearch

    
    def ask_question(self, question:str):
        return self.qa.invoke(question)
    
    def delete(self):
        self.pc.delete_index(self.index_name)

    

    

# TODO: add a list of llms and embedders - RK
# TODO: add device initialization
class LLM:
    def __init__(self, model_name:str='gpt2') -> None:
        if model_name == '':
            raise Exception("Please provide a valid model name.")
        
        if torch.backends.mps.is_available():
            print('Using MPS')
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            print('Using CUDA')
            self.device = torch.device("cuda")
        else:
            print('Using CPU')
            self.device = torch.device("cpu")
        

        # self.device = torch.device("cpu")

        if torch.cuda.device_count() > 1:
            distributed = True
        else:
            distributed = False


        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    
    def generate(self, 
                 text:str, 
                 max_new_tokens:int=50, 
                 num_return_sequences:int=1,
                 apply_tempplate:bool=False):
        """
        function to generate text from a given text.
        args:
            text: str, the text to generate from
            max_length: int, the maximum length of the generated text
            num_return_sequences: int, the number of sequences to return
        """
        encoded_input = self.tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences)
        # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)




# TODO: add device initialization
class embedder:
    def __init__(self, embedder:str='', chunk_size:int=30, overlap:int=10, docs:list='') -> None:
        if embedder == '':
            raise Exception("Please provide a valid embedder.")
        self.embedder = embedder
        self.model = SentenceTransformer(embedder, trust_remote_code=True)
        self.syn = synthetic()
        self.eval = eval_gen()
        self.docs = docs
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def embed_batch(self, sentences):
        """
        function to embed a batch of sentences.
        args:
            sentences: list of str, the sentences to embed
        """
        return self.model.encode(sentences)
    

    def embed(self, text):
        """
        function to embed a text. it first chunks it.
        args:
            text: str, the text to embed
            chunk_size: int, the size of the chunks to embed
        """
        chunks = chunk_text_by_words(text, self.chunk_size)
        embeddings = self.model.encode(chunks)
        return embeddings
    
    def embed_single(self, sentence):
        """
        function to embed a single sentence.
        args:
            sentence: str, the sentence to embed
        """
        return np.array([self.model.encode(sentence)])
    
    # TODO: time this
    def evaluate_fast(self, text:str='', ratio_of_shrink:float=0.01, save_gen_data:bool=False, save_path:str=''):
        """
        function to evaluate a text.
        args:
            text: str, the text to evaluate
        """
        if text == '' and self.docs == []:
            raise ValueError("text and doc list are both empty, need at least one to perform eval.")
        chunks = []
        questions = []
        answer_inds = []
        if text == '':
            print('text not provided, going thorugh documents')
            for doc in self.docs:
                text = read_pdf(doc)
                questions_q, answers_q, chunks_q, answer_inds_q = self.syn.create_qa_pair_single(text, chunk_size=self.chunk_size, overlap=self.overlap, ratio_of_shrink=ratio_of_shrink)
                chunks.append(chunks_q)
                questions.append(questions_q)
                answer_inds.append(answer_inds_q)
            # flatten the list of chunks, questions and answer_inds
            chunks = [item for sublist in chunks for item in sublist]
            questions = [item for sublist in questions for item in sublist]
            answer_inds = [item for sublist in answer_inds for item in sublist]
        else:
            questions, answers, chunks, answer_inds = self.syn.create_qa_pair_single(text, chunk_size=self.chunk_size, overlap=self.overlap, ratio_of_shrink=ratio_of_shrink)
        
        questions, answer_inds = shuffle(questions, answer_inds)
        
        res = self.eval.eval_embedder_with_pairs(questions, chunks, embedder=self.embedder, relevant_ids=answer_inds)

        if save_gen_data:
            dic = {'chunks': chunks, 'questions': questions, 'answer_inds': answer_inds}
            return res, dic
        else:
            return res

        

# TODO: add faiss for fast search
class QA_pipe:
    def __init__(self,
                 embedder_name:str='', 
                 llm_name:str='',
                 documents:list=[], 
                 cache_embeddings:bool=True, 
                 topk:int=5,
                 different_qa_embedder:bool=False,
                 embedder_name_q:str='',
                 chunk_size:int=20,
                 overlap:int=5) -> None:
        if embedder_name == '':
            raise Exception("Please provide a valid embedder.")
        
        if len(documents) == 0:
            raise Exception("Please provide a valid list of documents.")
        
        self.topk = topk
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.different_qa_embedder = different_qa_embedder
        
        self.embedder = embedder(embedder_name, chunk_size, overlap, documents)
        if different_qa_embedder:
            self.embedder_q = embedder(embedder_name_q)
            # assert if the output shape of embedder_q and embedder are the same
            if self.embedder_q.model.encode(['test']).shape != self.embedder.model.encode(['test']).shape:
                raise Exception("The output shape of the two embedders should be the same.")
        self.cache_embeddings = cache_embeddings

        # embed the documents on initialization
        self.meta_data = {'doc':[], 'chunks':[], 'embeddings':[], 'start':[], 'end':[]}
        if cache_embeddings:
            self.embed_documents(documents, chunk_size, overlap)

        # inintialize the LLM
        self.llm = LLM(llm_name)

    def embed_documents(self, documents, chunk_size=40, overlap=5):
        for doc in documents:
            text = read_pdf(doc)
            chunks, starts, ends = chunk_text_by_words(text, chunk_size, return_start_end=True, overlap=overlap)
            embeddings = self.embedder.embed_batch(chunks)
            self.meta_data['doc'].append(doc)
            self.meta_data['chunks'].append(chunks)
            self.meta_data['embeddings'].append(embeddings)
            self.meta_data['start'].append(starts)
            self.meta_data['end'].append(ends)
        self.meta_data['embeddings'] = np.vstack(self.meta_data['embeddings'])
        # flatten the list of chunks
        self.meta_data['chunks'] = [item for sublist in self.meta_data['chunks'] for item in sublist]


    def embed(self, text, chunk_size=100):
        """
        function to embed a text.
        args:
            text: str, the text to embed
            chunk_size: int, the size of the chunks to embed
        """
        chunks = chunk_text_by_words(text, chunk_size)
        embeddings = self.embedder.embed(chunks)
        return embeddings
    
    def get_inds(self, question):
        """
        function to ask a question from a text.
        args:
            question: str, the question to ask
            text: str, the text to ask the question from
            chunk_size: int, the size of the chunks to embed
        """
        if self.different_qa_embedder:
            question_embedding = self.embedder_q.embed_single(question)
        else:
            question_embedding = self.embedder.embed_single(question)

        if self.cache_embeddings:
            scores = cos_sim(question_embedding, self.meta_data['embeddings'])
        else:
            self.embed_documents(self.documents, self.chunk_size, self.overlap)
            question_embedding = self.embedder.embed_single(question)
            scores = cos_sim(question_embedding, self.meta_data['embeddings'])
        
        inds_per_q = torch.topk(scores, self.topk).indices
        return scores, inds_per_q.numpy()
    
    def ask_question(self, 
                     questions:list=[],
                     return_relevant_chunks:bool=False,):
        if not isinstance(questions, list):
            raise TypeError(f"""Expected 'questions' to be a list, but got {type(questions).__name__} instead.
                            If you have a single question, please provide it as a list with a single element.""")
    
        answers = []
        relevant_chunks_all = []
        for question in questions:
            scores, inds_per_q = self.get_inds(question)
            
            print(f'inds: {inds_per_q}')
            for inds in inds_per_q:
                relevant_chunks = [self.meta_data['chunks'][ind] for ind in inds]
                relevant_chunks_all.append(relevant_chunks)
                string = f'Question: {question}\n'
                for i, chunk in enumerate(relevant_chunks):
                    string += f'Chunk {i+1}: {chunk}\n'
                string += 'Answer: '
                answer = self.llm.generate(string, max_new_tokens=50, num_return_sequences=1)
            answers.append(answer)
        if return_relevant_chunks:
            return answers, relevant_chunks_all
        else:
            return answers
        

    
    

if __name__ == "__main__":
    text = """Gothic resistance revived however, and on 17 December 546, the Ostrogoths under Totila recaptured and sacked Rome.[57] 
    Belisarius soon recovered the city, but the Ostrogoths retook it in 549. Belisarius was replaced by Narses, who captured Rome 
    from the Ostrogoths for good in 552, ending the so-called Gothic Wars which had devastated much of Italy. 
    The continual war around Rome in the 530s and 540s left it in a state of total disrepair 
     near-abandoned and desolate with much of its lower-lying parts turned into unhealthy marshes as the drainage systems were neglected 
     and the Tiber's embankments fell into disrepair in the course of the latter half of the 6th century.[58] Here, malaria developed. 
     The aqueducts, except for one, were not repaired. The population, without imports of grain and oil from Sicily, 
     shrank to less than 50,000 concentrated near the Tiber and around the Campus Martius, abandoning those districts without water supply. 
     There is a legend, significant though untrue, that there was a moment where no one remained living in Rome.[citation needed]

    Justinian I provided grants for the maintenance of public buildings, aqueducts and bridges—though, being mostly drawn from an 
    Italy dramatically impoverished by the recent wars, these were not always sufficient. He also styled himself 
    the patron of its remaining scholars, orators, physicians and lawyers in the stated hope that eventually more youths 
    would seek a better education. After the wars, the Senate was theoretically restored, but under the supervision of the urban 
    prefect and other officials appointed by, and responsible to, the Eastern Roman authorities in Ravenna."""
    # emb = embedder('nomic-ai/nomic-embed-text-v1.5')
    # chunks = emb.chunk_text(text, 100, 20)
    # embeds = emb.embed(text, 50)
    # print(type(emb))
    # documents=['/Users/alikavoosi/Desktop/DEMO/d5.pdf',
    #            '/Users/alikavoosi/Desktop/DEMO/d4.pdf',
    #            '/Users/alikavoosi/Desktop/DEMO/d3.pdf',
    #            '/Users/alikavoosi/Desktop/DEMO/d2.pdf',
    #            '/Users/alikavoosi/Desktop/DEMO/d1.pdf']
    # qa = QA_pipe(embedder_name='nomic-ai/nomic-embed-text-v1.5', documents=documents, cache_embeddings = True, llm_name='openai-community/gpt2-large', topk=2, chunk_size=40, overlap=5)
    
    # resp = qa.ask_question(["What company are the documents about?"])
    # print(resp)
    # print('\n----------\n')
    # print(qa.meta_data)
    # llm = LLM('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    # print(llm.generate('what is your puepose in a few words? answer:', max_length=90, num_return_sequences=1))
    # path = '/Users/alikavoosi/Desktop/DEMO/newpdf.pdf'
    # pdb = piencone_db(index_name='test-index', namespace='test-namespace',path=path, mode='pdf', text=text)
    # print(pdb.ask_question('what is the main idea behind the article?'))
    # pdb.delete()
    pdb = pincecone_db_man()
    pdb.add_data(['paris is the capital of france','berlin is the capital of germany','weather is so bad today'], log_local=True)
    result = pdb.ask(question='capital of france')
    print(result)
    # print(pdb.index.describe_index_stats()['total_vector_count'])
