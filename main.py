import os
import datetime
import logging
import sys
import time
import json

import openai
import PyPDF2
import boto3

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding

dynamodb = boto3.resource('dynamodb',
    region_name='eu-west-1',
    aws_access_key_id=os.getenv('aws_access_key_id'),
    aws_secret_access_key=os.getenv('aws_secret_access_key')
)

table = dynamodb.Table('RAGproject')


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper

@timer_decorator
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

@timer_decorator
def chunking(documents, chunk_method = ''):
    chunked_docs = []
    for doc in documents:
        
        # using different chunking params
        if chunk_method == 'paragraph':
            paragraphs = doc.text.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():  # Ignore empty paragraphs
                    chunked_doc = Document(text=paragraph, metadata=doc.metadata)
                    chunked_docs.append(chunked_doc)
        else:
            chunk_size = 1000
            text = doc.text
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                chunked_doc = Document(text=chunk, metadata=doc.metadata)
                chunked_docs.append(chunked_doc)

    return chunked_docs

def query(query, embed_model, chunking_method):
    openai.api_key = os.getenv('openai_key')
    documents = SimpleDirectoryReader("data").load_data()

    chunked_documents = chunking(documents)

    index = VectorStoreIndex.from_documents(chunked_documents, embedding=embed_model) if embed_model else VectorStoreIndex.from_documents(chunked_documents)

    ### ? print(embed_model)

    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    print(response.__dict__)

    return response_to_dict(response)

def response_to_dict(response):
    return {
        'response': response.response,
        'metadata': response.metadata
    }

@timer_decorator
def process_queries(query_function, model, chunking_method, embed_model = None):

    os.makedirs('output', exist_ok=True)

    with open('queries.txt', 'r') as queries_file, open(f'output/{model}_output.txt', 'w') as output_file:

        queries = queries_file.readlines()

        for query_text in queries:

            query_text = query_text.strip()

            response = query_function(query_text, embed_model, chunking_method)

            output_file.write(str(response) + '\n\n')