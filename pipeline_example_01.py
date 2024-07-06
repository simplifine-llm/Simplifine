from pipeline import *
from eval import *
from annotaion import *
from utils import read_pdf
from train_engine import hf_finetune_embedder_positive
from sklearn.utils import shuffle
import asyncio


chunk_size=50
overlap=10
ratio_of_shrink=0.01

documents = ["/Users/alikavoosi/Desktop/DEMO/newpdf.pdf"]
embedder_name = 'sentence-transformers/sentence-t5-large'

for doc in documents:
    text = read_pdf(doc)

qa = QA_pipe(embedder_name=embedder_name, documents=documents, cache_embeddings = True, llm_name='openai-community/gpt2-large', topk=5, chunk_size=chunk_size, overlap=overlap)

eval_result = qa.embedder.evaluate_fast(ratio_of_shrink=ratio_of_shrink)
print(f'eval_result: {eval_result}')

# queries = ["What is the main idea of the article?", "what companies are mentioned in the article?", "who is the author of the article?", "what is the publication date of the article?"]
# resp, chunks = qa.ask_question(queries, return_chunks=True)

# # evaluting the responses
# eval = eval_gen()
# result = eval.sync_embed_eval_per_ctx(queries=queries, chunks=chunks)
# print(f'resutl \n ========== \n: {result}')

# generating synthetic dataset for the document
# syn = synthetic()
# for doc in documents:
#     text = read_pdf(doc)
    # questions, answers, chunks, answer_inds = asyncio.run(syn.create_qa_pair_single_async(text, chunk_size=chunk_size, overlap=overlap, ratio_of_shrink=ratio_of_shrink))
    # print(f'\n==============\n questions: {questions} \n answers: {answers} \n answer_inds: {answer_inds} \n==============\n')
    # syn.QA_gen_batch(text, chunk_size, overlap, shrink=True, ratio_of_shrink=ratio_of_shrink)

# questions = ['a','b','c','d','e','f','g','h','i','j']
# chunks = ['a','b','c','d','e','f','g','h','i','j','k','a','b','c','d','e','f','g','h','i','j','k']
# answer_inds = list(np.arange(len(questions)))

# questions, answer_inds = shuffle(questions, answer_inds)

# res = eval.eval_embedder_with_pairs(questions, chunks, embedder=embedder_name, relevant_ids=answer_inds)

# hf_finetune_embedder_positive(model_name=embedder_name, questions=questions, docs = chunks, relevant_ids=answer_inds, do_split=True,
#                               train_split_ratio=0.5)
