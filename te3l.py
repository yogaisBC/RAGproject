import os
import datetime
import logging
import sys

from main import *

now = datetime.datetime.now()

timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

filename = f"logs/log_{timestamp}.txt"

logging.basicConfig(filename=filename, level=logging.DEBUG)

load_dotenv()

embed_model = OpenAIEmbedding(model="text-embedding-3-large")

pdf_to_text('law.pdf', 'data/law.txt')

process_queries(query, 'text-embedding-3-large', embed_model)
