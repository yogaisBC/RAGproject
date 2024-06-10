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

embed_model = OpenAIEmbedding(model="text-embedding-3-large")

def docs_to_dynamodb(embed_model):
    try:
        documents = SimpleDirectoryReader("data").load_data()
        chunked_documents = chunking(documents)

        for doc in chunked_documents:
            # Split document into nodes
            nodes = SentenceSplitter().get_nodes_from_documents([doc])
            
        storage_context = StorageContext.from_defaults(
            docstore=DynamoDBDocumentStore.from_table_name(table_name=TABLE_NAME, session=session),
            index_store=DynamoDBIndexStore.from_table_name(table_name=TABLE_NAME, session=session),
            vector_store=DynamoDBVectorStore.from_table_name(table_name=TABLE_NAME, session=session),
        )
            
        storage_context.docstore.add_documents(nodes)

        summary_index = SummaryIndex(nodes, storage_context=storage_context)
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context,embed_model=embed_model)
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