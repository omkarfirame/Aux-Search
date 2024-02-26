import streamlit as st
from AuxSearch.components.PDFTextExtractor import PDFConverter
from AuxSearch.components.TextChunkerEmbedder import TextProcessor
from langchain_community.vectorstores.faiss import FAISS
from AuxSearch.constants import *
from AuxSearch.components.Model import ResponseGenerator

pdfconverter = PDFConverter()
textprocessor = TextProcessor()
responsegen = ResponseGenerator()




import streamlit as st

def main():
    st.title("Question Answering System")

    # File upload
    uploaded_file = st.file_uploader("Upload a file", type=["pdf"])
    question = st.text_input("Enter your question")

    if uploaded_file and question:
        submit_button = st.button("Submit")
        if submit_button:
            raw_text = pdfconverter.pdf_to_text(uploaded_file)
            text_chunks = textprocessor.get_chunks(raw_text)
            textprocessor.get_vector_store(text_chunks)


            # Question input
            

            if st.button("Get Answer"):
                # Initialize UserInput instance
                rag_chain = responsegen.get_response()
                # Display the answer
                st.subheader("Answer:")
                st.write(rag_chain.invoke(question))
        
if __name__ == "__main__":
    main()

