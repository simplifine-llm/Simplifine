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
import PyPDF2


def read_jsonl(file_path:str):
    """
    Reads a JSONL (JSON Lines) file and returns a list of JSON objects.

    Parameters:
    file_path (str): The path to the JSONL file.

    Returns:
    list: A list of JSON objects read from the file.

    Raises:
    FileNotFoundError: If the file at file_path does not exist.
    json.JSONDecodeError: If a line in the file is not a valid JSON object.

    Example:
    >>> json_objects = read_jsonl("data.jsonl")
    >>> print(json_objects)
    [{'key1': 'value1'}, {'key2': 'value2'}, ...]

    Notes:
    - Each line in the file should be a valid JSON object.
    - The file should be encoded in UTF-8.
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
    text (str): The input text to be chunked. Default is an empty string.
    chunk_length (int): The number of words in each chunk. Must be greater than 0.
    overlap (int): The number of words each chunk should overlap with the previous chunk. Must be 0 or greater.
    verbose (bool): If True, prints additional information about the chunking process. Default is True.
    return_start_end (bool): If True, returns the start and end indices of each chunk along with the chunks. Default is False.

    Returns:
    list: A list of text chunks if return_start_end is False.
    tuple: A tuple containing three lists (chunks, starts, ends) if return_start_end is True.

    Raises:
    ValueError: If chunk_length is less than or equal to overlap, or if chunk_length is less than or equal to 0, or if overlap is less than 0.

    Example:
    >>> chunks = chunk_text_by_words("This is a sample text to be chunked.", chunk_length=4, overlap=2)
    >>> print(chunks)
    ['This is a sample', 'sample text to', 'to be chunked']
    
    >>> chunks, starts, ends = chunk_text_by_words("This is a sample text to be chunked.", chunk_length=4, overlap=2, return_start_end=True)
    >>> print(chunks)
    ['This is a sample', 'sample text to', 'to be chunked']
    >>> print(starts)
    [0, 2, 4]
    >>> print(ends)
    [4, 6, 8]
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
    Reads a PDF file and returns the extracted text.

    Parameters:
    path (str): The file path to the PDF file to be read.

    Returns:
    str: The extracted text from the PDF file.

    Example:
    >>> text = read_pdf("example.pdf")
    >>> print(text)
    This is the extracted text from the PDF file.
    """

    text = ""
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text