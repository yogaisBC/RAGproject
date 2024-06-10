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

from query import query

nest_asyncio.apply()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

# Load environment variables from .env file
load_dotenv()

TABLE_NAME = os.getenv('DYNAMODB_TABLE_NAME')
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
OPENAI_KEY = os.getenv('OPENAI_KEY')


#setup dynamodb, credentials
def dynamodb_setup():
    try:
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        dynamodb = session.client('dynamodb')
        region = dynamodb.meta.region_name
        logger.info(f"DynamoDB client setup successfully in region: {region}")
        return dynamodb
    except Exception as e:
        logger.error(f"Error setting up DynamoDB client: {e}")
        raise

### ! go to storing
def docs_to_dynamodb(embed_model):
    try:
        documents = SimpleDirectoryReader("data").load_data()
        chunked_documents = chunking(documents)

        for doc in chunked_documents:
            # Split document into nodes
            nodes = SentenceSplitter().get_nodes_from_documents([doc])
            
        storage_context = StorageContext.from_defaults(
            docstore=DynamoDBDocumentStore.from_table_name(table_name=TABLE_NAME),
            index_store=DynamoDBIndexStore.from_table_name(table_name=TABLE_NAME),
            vector_store=DynamoDBVectorStore.from_table_name(table_name=TABLE_NAME),
        )
            
        storage_context.docstore.add_documents(nodes)

        summary_index = SummaryIndex(nodes, storage_context=storage_context)
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
        keyword_table_index = SimpleKeywordTableIndex(
            nodes, storage_context=storage_context
        )

        list_id = summary_index.index_id
        vector_id = vector_index.index_id
        keyword_id = keyword_table_index.index_id

        os.makedirs('id', exist_ok=True)

        with open('id/list_id.txt', 'w') as f:
            f.write(str(list_id))
        with open('id/vector_id.txt', 'w') as f:
            f.write(str(vector_id))
        with open('id/keyword_id.txt', 'w') as f:
            f.write(str(keyword_id))

        logger.info(f"Documents added to DynamoDB: {nodes}")

    except Exception as e:
        logger.error(f"Error in docs_to_dynamodb: {e}")
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

if __name__ == '__main__':
    try:
        openai.api_key = OPENAI_KEY
        dynamodb = dynamodb_setup()
        pdf_to_text('pdf','data')
        embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        docs_to_dynamodb(embed_model)
        query(TABLE_NAME, embed_model)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
