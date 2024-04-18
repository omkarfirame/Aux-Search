import streamlit as st
from AuxSearch.components.PDFTextExtractor import PDFConverter
from AuxSearch.components.TextChunkerEmbedder import TextProcessor
# from AuxSearch.constants import *
from AuxSearch.components.Model import ResponseGenerator
from flask import Flask, render_template, request
import openai

pdfconverter = PDFConverter()
textprocessor = TextProcessor()
responsegen = ResponseGenerator()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer_question():
    # Get user inputs
    question = request.form['question']
    pdf_file = request.files['pdf_file']

    # Process PDF file
    raw_text = pdfconverter.pdf_to_text(pdf_file)
    text_chunks = textprocessor.get_chunks(raw_text)
    textprocessor.get_vector_store(text_chunks)
    rag_chain = responsegen.get_response()

    return render_template('answer.html', question=question, answer=rag_chain.invoke(question))


if __name__ == '__main__':
    app.run(debug=True)
