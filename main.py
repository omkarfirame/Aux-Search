import streamlit as st
from AuxSearch.components.PDFTextExtractor import PDFConverter
from AuxSearch.components.TextChunkerEmbedder import TextProcessor
from AuxSearch.components.Model import ChainLoader
from langchain.embeddings import OllamaEmbeddings 
from langchain_community.vectorstores import FAISS
from AuxSearch.constants import *

pdfconverter = PDFConverter()
textprocessor = TextProcessor()
chains = ChainLoader()


def user_input(user_question):
    embeddings = OllamaEmbeddings()
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = chains.get_chains()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

st.set_page_config("Chat PDF")
st.header("Chat with PDF using LLAMA2 ")

user_question = st.text_input("Ask a Question from the PDF Files")

if user_question:
    user_input(user_question)

with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = pdfconverter.pdf_to_text(pdf_docs)
            text_chunks = textprocessor.get_chunks(raw_text)
            textprocessor.get_vector_store(text_chunks)
            st.success("Done")