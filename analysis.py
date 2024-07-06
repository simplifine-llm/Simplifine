from openai import OpenAI
import os
import json
import torch
import numpy as np
from text_chunker import TextChunker
import faiss


class embedding:
    def __init__(self):
        self.client = OpenAI(
        api_key='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',
        )
        self.model_name = 'text-embedding-3-large'
        self.script_path = os.path.dirname(os.path.abspath(__file__))

    def openai_embedding_single(self, text):
        """
        get the embedding of a single text
        args:
        text: str, the text to get the embedding for
        returns:
        embedding: list, the embedding of the text
        """
        text = text.replace("\n", " ")
        array = self.client.embeddings.create(input = [text], model=self.model_name).data[0].embedding
        return np.array(array)
    




if __name__ == '__main__':
    emb = embedding()
    with open('/Users/alikavoosi/Desktop/COMPSOFT/exampel_text/PG_essay.txt', 'r') as file:
        text = file.read()
    chunks = emb.chunk_text_by_words(text, 1000)
    embd = emb.openai_embedding_single(chunks)
    print(embd.shape)
    