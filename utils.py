# from text_chunker import TextChunker
import json
import PyPDF2


def read_jsonl(file_path:str):
    """
    Reads a JSONL file and returns a list of JSON objects.
    
    :param file_path: The path to the JSONL file.
    :return: A list of JSON objects.
    """
    json_objects = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_objects.append(json.loads(line.strip()))
    return json_objects

def chunk_text_by_words(text:str='', 
                        chunk_length:int=100, 
                        overlap:int=10, 
                        verbose:bool=True,
                        return_start_end:bool=False):
    """
    Splits text into chunks of specified word length with a specified overlap of words.

    Parameters:
    text (str): The input text to be chunked.
    chunk_length (int): The number of words in each chunk.
    overlap (int): The number of words each chunk should overlap with the previous chunk.

    Returns:
    list: A list of text chunks.
    """
    if chunk_length <= 0:
        raise ValueError("chunk_length must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be 0 or greater")
    if chunk_length <= overlap:
        raise ValueError("chunk_length must be greater than overlap")
    
    words = text.split()
    chunks = []
    start = 0
    starts = []
    ends = []
    while start < len(words):
        end = start + chunk_length
        starts.append(start)
        ends.append(end)
        chunk = words[start:end]
        chunks.append(' '.join(chunk))
        start += chunk_length - overlap
    if return_start_end:
        return chunks, starts, ends
    else:
        return chunks
    
def read_pdf(path):
    """
    Function to read a pdf file and return the text.
    """
    text = ""
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text




# if __name__ == '__main__':
#     path = '/Users/alikavoosi/Desktop/COMPSOFT/Src/batch requests/batch_request_train_2.jsonl'
#     data = read_jsonl(path)
#     print(data[0]['body']['messages'][-1]['content'])
    