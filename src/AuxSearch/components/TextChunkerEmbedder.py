
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings import OllamaEmbeddings 
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class TextProcessor:
    def __init__(self):
        pass
    
    def get_chunks(self, text):
        """
        parameter: text from pdf to text
        return: text in chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(self, chunks):
        """
        parameter: text chunks from get_chunks function
        saves vectors to faiss vector database 
        """
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector = FAISS.from_texts(chunks, embeddings)
        vector.save_local("faiss_index")
