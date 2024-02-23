from PyPDF2 import PdfReader

class PDFConverter:
    def __init__(self):
        pass
    
    def pdf_to_text(self, pdf_files):
        """
        parameter: pdf_files (list of strings)
        return: single text file (string)
        """

        text = ""
        reader = PdfReader(pdf_files)
        for page in reader.pages:
            text += page.extract_text()
        return text
