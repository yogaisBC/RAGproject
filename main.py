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

    pdfFileObj = open(pdf_path, 'rb')

    pdfReader = PyPDF2.PdfReader(pdfFileObj)

    num_pages = len(pdfReader.pages)

    text = ""

    for page in range(num_pages):
        pageObj = pdfReader.pages[page]
        text += pageObj.extract_text()

    pdfFileObj.close()

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