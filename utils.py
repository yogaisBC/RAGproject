import PyPDF2

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
