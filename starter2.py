import os
import datetime
import logging
import sys

import openai
import PyPDF2

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

now = datetime.datetime.now()

timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

filename = f"logs/log_{timestamp}.txt"

logging.basicConfig(filename=filename, level=logging.DEBUG)

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