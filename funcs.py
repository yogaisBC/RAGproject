import os
import datetime
import logging
import sys
import time
import json
from decimal import Decimal

import openai
import PyPDF2
import boto3

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, load_index_from_storage, StorageContext
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.dynamodb import DynamoDBVectorStore

def dynamodb_setup():

    dynamodb = boto3.client(
        'dynamodb',
        region_name='eu-west-2',
        aws_access_key_id=os.getenv('aws_access_key_id'),
        aws_secret_access_key=os.getenv('aws_secret_access_key')
    )

    table_name='RAG'

    dynamodb_vector_store = DynamoDBVectorStore.from_table_name(
        table_name=table_name
        
        )

    storage_context = StorageContext.from_defaults(vector_store=dynamodb_vector_store)

def dynamodb_entry(entry):
    #put items into dynamodb
    

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper

@timer_decorator
def pdf_to_text(pdf_directory, txt_directory):
    files = os.listdir(pdf_directory)
    for file in files:
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, file)
            txt_path = os.path.join(txt_directory, file.replace('.pdf', '.txt'))

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

    index = text_to_embeddings(embed_model)

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
def text_to_embeddings(embed_model):

    dynamodb_setup()
    
    documents = SimpleDirectoryReader("data").load_data()

    chunked_documents = chunking(documents)

    index = VectorStoreIndex.from_documents(chunked_documents, embedding=embed_model) if embed_model else VectorStoreIndex.from_documents(chunked_documents)

    return index

@timer_decorator
def process_queries(query_function, model, chunking_method, embed_model = None):

    os.makedirs('output', exist_ok=True)

    with open('queries.txt', 'r') as queries_file, open(f'output/{model}_output.txt', 'w') as output_file:

        queries = queries_file.readlines()

        for query_text in queries:

            query_text = query_text.strip()

            response = query_function(query_text, embed_model, chunking_method)

            output_file.write(str(response) + '\n\n')

if __name__ == '__main__':
    load_dotenv()

    dynamodb_setup()

    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    process_queries(query, 'text-embedding-3-large', 'paragraph', embed_model)