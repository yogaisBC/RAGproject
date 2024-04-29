import os
import openai

import PyPDF2
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

def pdf_to_text(pdf_path, txt_path):
    # Open file path
    pdfFileObj = open(pdf_path, 'rb')

    # Create a PDF reader object
    pdfReader = PyPDF2.PdfReader(pdfFileObj)

    # Get the number of pages in PDF file
    num_pages = len(pdfReader.pages)

    # Initialize a text variable
    text = ""

    # Extract text from each page
    for page in range(num_pages):
        pageObj = pdfReader.pages[page]
        text += pageObj.extract_text()

    # Close the PDF file object
    pdfFileObj.close()

    # Write the extracted text to a .txt file
    with open(txt_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text)

# Usage
pdf_to_text('law.pdf', 'data/law.txt')

openai.api_key = os.getenv('openai_key')
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, model='text-embeddings-large')

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

print(f"Loaded {len(documents)} documents.")

print(response)