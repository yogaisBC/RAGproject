import logging
import sys
import os
import time
import json
from decimal import Decimal

import openai
import PyPDF2
import boto3
import nest_asyncio

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext, SummaryIndex, SimpleKeywordTableIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.storage.docstore.dynamodb import DynamoDBDocumentStore
from llama_index.vector_stores.dynamodb import DynamoDBVectorStore
from llama_index.storage.index_store.dynamodb import DynamoDBIndexStore
from llama_index.core.node_parser import SentenceSplitter

nest_asyncio.apply()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

def dynamodb_setup():
    try:
        dynamodb = boto3.client(
            'dynamodb',
            aws_access_key_id=os.getenv('aws_access_key_id'),
            aws_secret_access_key=os.getenv('aws_secret_access_key'),
            region_name=os.getenv('aws_region')  # Ensure the region is set in your .env file
        )
        region = dynamodb.meta.region_name
        logger.info(f"DynamoDB client setup successfully in region: {region}")
        return dynamodb
    except Exception as e:
        logger.error(f"Error setting up DynamoDB client: {e}")
        raise

def text_to_embeddings(embed_model):
    try:
        documents = SimpleDirectoryReader("data").load_data()
        chunked_documents = chunking(documents)

        for doc in chunked_documents:
            # Split document into nodes
            nodes = SentenceSplitter().get_nodes_from_documents([doc])
            
        storage_context = StorageContext.from_defaults(
            docstore=DynamoDBDocumentStore.from_table_name(table_name='RAG'),
            index_store=DynamoDBIndexStore.from_table_name(table_name='RAG'),
            vector_store=DynamoDBVectorStore.from_table_name(table_name='RAG'),
        )
            
        storage_context.docstore.add_documents(nodes)
        logger.info(f"Documents added to DynamoDB: {nodes}")

        index = VectorStoreIndex.from_documents(chunked_documents, embedding=embed_model) if embed_model else VectorStoreIndex.from_documents(chunked_documents)
        
        return index
    except Exception as e:
        logger.error(f"Error in text_to_embeddings: {e}")
        raise

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper

@timer_decorator
def pdf_to_text(pdf_directory, txt_directory):
    try:
        files = os.listdir(pdf_directory)
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, file)
                txt_path = os.path.join(txt_directory, file.replace('.pdf', '.txt'))

                with open(pdf_path, 'rb') as pdfFileObj:
                    pdfReader = PyPDF2.PdfReader(pdfFileObj)
                    num_pages = len(pdfReader.pages)

                    text = ""
                    for page in range(num_pages):
                        pageObj = pdfReader.pages[page]
                        text += pageObj.extract_text()

                with open(txt_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(text)

                logger.info(f"Converted {file} to text.")
    except Exception as e:
        logger.error(f"Error in pdf_to_text: {e}")
        raise

@timer_decorator
def chunking(documents, chunk_method=''):
    try:
        chunked_docs = []
        for doc in documents:
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

        logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks.")
        return chunked_docs
    except Exception as e:
        logger.error(f"Error in chunking: {e}")
        raise

@timer_decorator
def query(query, embed_model, chunking_method):
    try:
        openai.api_key = os.getenv('openai_key')

        index = text_to_embeddings(embed_model)

        query_engine = index.as_query_engine()
        response = query_engine.query(query)

        logger.info(f"Query response: {response.__dict__}")
        return response
    except Exception as e:
        logger.error(f"Error in query: {e}")
        raise

@timer_decorator
def process_queries(query_function, model, chunking_method, embed_model=None):
    try:
        os.makedirs('output', exist_ok=True)

        with open('queries.txt', 'r') as queries_file, open(f'output/{model}_output.txt', 'w') as output_file:
            queries = queries_file.readlines()

            for query_text in queries:
                query_text = query_text.strip()
                response = query_function(query_text, embed_model, chunking_method)
                output_file.write(str(response) + '\n\n')

        logger.info(f"Processed queries with model {model}.")
    except Exception as e:
        logger.error(f"Error in process_queries: {e}")
        raise

if __name__ == '__main__':
    try:
        load_dotenv()
        dynamodb = dynamodb_setup()
        embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        process_queries(query, 'text-embedding-3-large', 'paragraph', embed_model)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
