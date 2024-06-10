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
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

openai.api_key = os.getenv('openai_key')

def dynamodb_setup():
    dynamodb = boto3.client(
        'dynamodb',
        region_name='ca-central-1',
        aws_access_key_id=os.getenv('aws_access_key_id'),
        aws_secret_access_key=os.getenv('aws_secret_access_key')
    )

    table_name = 'RAG'

    storage_context = StorageContext.from_defaults(
        docstore=DynamoDBDocumentStore.from_table_name(table_name=table_name),
        vector_store=DynamoDBVectorStore.from_table_name(table_name=table_name),
    )

    return storage_context

def chunking(documents, chunk_size=1000):
    chunked_docs = []
    for doc in documents:
        text = doc.text
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            chunked_doc = Document(text=chunk, metadata=doc.metadata)
            chunked_docs.append(chunked_doc)
    return chunked_docs

def text_to_embeddings(embed_model):
    storage_context = dynamodb_setup()
    
    documents = SimpleDirectoryReader("data").load_data()
    chunked_documents = chunking(documents)

    for doc in chunked_documents:
        # Split document into nodes
        nodes = SentenceSplitter().get_nodes_from_documents([doc])
        
        # Create indices from nodes
        summary_index = SummaryIndex(nodes, storage_context=storage_context)
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
        keyword_table_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)
        
        # Add nodes to the document store
        storage_context.docstore.add_documents(nodes)

    index = VectorStoreIndex.from_documents(chunked_documents, embedding=embed_model) if embed_model else VectorStoreIndex.from_documents(chunked_documents)
    
    return index

if __name__ == '__main__':
    embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")
    index = text_to_embeddings(embed_model)
    print("Index created successfully.")
