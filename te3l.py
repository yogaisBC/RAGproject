import os
import datetime
import logging
import sys

import openai
import PyPDF2

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

from utils import pdf_to_text

now = datetime.datetime.now()

timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

filename = f"logs/log_{timestamp}.txt"

logging.basicConfig(filename=filename, level=logging.DEBUG)

load_dotenv()

# Load the OpenAI embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

### ? print(embed_model)


def query(query):
    openai.api_key = os.getenv('openai_key')
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents, embedding=embed_model)

    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    return response

pdf_to_text('law.pdf', 'data/law.txt')

def process_queries(query_function):
    with open('queries.txt', 'r') as queries_file, open('output.txt', 'w') as output_file:
        # Read the queries
        queries = queries_file.readlines()

        # Iterate over the queries
        for query_text in queries:
            # Remove trailing newline characters
            query_text = query_text.strip()

            # Run the query
            response = query_function(query_text)

            # Write the response to the output file
            output_file.write(str(response) + '\n\n')