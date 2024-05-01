import os
import datetime
import logging
import sys

import openai
import PyPDF2

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

def pdf_to_text(pdf_path, txt_path):
    # Open file path
    pdfFileObj = open(pdf_path, 'rb')

    # Create a PDF reader object
    pdfReader = PyPDF2.PdfReader(pdfFileObj)

    # Get the number of pages in PDF file
    num_pages = len(pdfReader.pages)

    # Initialize a text variable
    text = ""

    # Extract text from each page
    for page in range(num_pages):
        pageObj = pdfReader.pages[page]
        text += pageObj.extract_text()

    # Close the PDF file object
    pdfFileObj.close()

    # Write the extracted text to a .txt file
    with open(txt_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text)

def query(query, embed_model):
    openai.api_key = os.getenv('openai_key')
    documents = SimpleDirectoryReader("data").load_data()

    index = VectorStoreIndex.from_documents(documents, embedding=embed_model) if embed_model else VectorStoreIndex.from_documents(documents)

    print(embed_model)

    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    return response

def process_queries(query_function, model, embed_model = None):

    os.makedirs('output', exist_ok=True)

    with open('queries.txt', 'r') as queries_file, open(f'output/{model}_output.txt', 'w') as output_file:

        queries = queries_file.readlines()

        for query_text in queries:

            query_text = query_text.strip()

            response = query_function(query_text, embed_model)

            output_file.write(str(response) + '\n\n')