import logging
import sys
import os
import time
import json
from decimal import Decimal

import openai
import boto3
import nest_asyncio

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext, SummaryIndex, SimpleKeywordTableIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.storage.docstore.dynamodb import DynamoDBDocumentStore
from llama_index.vector_stores.dynamodb import DynamoDBVectorStore
from llama_index.storage.index_store.dynamodb import DynamoDBIndexStore
from llama_index.core import load_index_from_storage

nest_asyncio.apply()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

def query(table_name, embed_model):
    try:
        with open('id/list_id.txt', 'r', encoding='utf-8') as file:
            list_id = file.read()
        with open('id/vector_id.txt', 'r', encoding='utf-8') as file:
            vector_id = file.read()
        with open('id/keyword_id.txt', 'r', encoding='utf-8') as file:
            keyword_id = file.read()

        storage_context = StorageContext.from_defaults(
            docstore=DynamoDBDocumentStore.from_table_name(table_name=table_name),
            index_store=DynamoDBIndexStore.from_table_name(table_name=table_name),
            vector_store=DynamoDBVectorStore.from_table_name(table_name=table_name),
        )

        summary_index = load_index_from_storage(
            storage_context=storage_context, index_id=list_id
        )
        keyword_table_index = load_index_from_storage(
            storage_context=storage_context, index_id=keyword_id
        )

        vector_index = load_index_from_storage(
            storage_context=storage_context, index_id=vector_id,embed_model=embed_model
        )

        os.makedirs('output', exist_ok=True)

        query_engine = vector_index.as_query_engine()
        with open('queries.txt', 'r') as queries_file, open(f'output/output.txt', 'w') as output_file:
            queries = queries_file.readlines()
        
            for query_text in queries:
                query_text = query_text.strip()
                response = query_engine.query(query_text)

                logger.info(f"Query response: {response.__dict__}")

                output_file.write(str(response) + '\n\n')

    except Exception as e:
        logger.error(f"Error in query: {e}")
        raise
