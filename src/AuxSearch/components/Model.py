from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from AuxSearch.constants import *
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st

class ResponseGenerator:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = FAISS.load_local("faiss_index", self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", k=3)

        self.prompt_template = """
            Answer the question to the point without description. If answer not found in context reply "Answer Not found"\n\n
            Context:\n {context}?\n
            Question: \n{question}\n

            Answer:
        """
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["context", "question"])

        self.rag_chain = (
            {
                "context": self.retriever,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def get_response(self):
        return self.rag_chain


