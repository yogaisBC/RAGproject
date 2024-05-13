import os
import datetime
import logging
import sys

from funcs import *

now = datetime.datetime.now()

timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

if not os.path.exists('logs'):
    os.makedirs('logs')

filename = f"logs/log_{timestamp}.txt"

logging.basicConfig(filename=filename, level=logging.DEBUG)

load_dotenv()

embed_model = OpenAIEmbedding(model="text-embedding-3-large")

pdf_to_text('pdf/', 'data/')

process_queries(query, 'text-embedding-3-large', 'paragraph', embed_model)
