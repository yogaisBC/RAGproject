import os
import datetime
import logging
import sys

import openai
import PyPDF2

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from utils import pdf_to_text

now = datetime.datetime.now()

timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

filename = f"logs/log_{timestamp}.txt"

logging.basicConfig(filename=filename, level=logging.DEBUG)

load_dotenv()

def query(query):
    openai.api_key = os.getenv('openai_key')
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    print(f"Loaded {len(documents)} documents.")

    print(response)


pdf_to_text('law.pdf', 'data/law.txt')

query("Are there any specific limitations on the quantity of controlled substances that can be prescribed or dispensed at one time in Florida?")