import os
import datetime
import logging
import sys

from funcs import *

now = datetime.datetime.now()

timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")   

filename = f"logs/log_{timestamp}.txt"

logging.basicConfig(filename=filename, level=logging.DEBUG)

load_dotenv()

pdf_to_text('law.pdf', 'data/law.txt')

process_queries(query, 'default', 'paragraph')