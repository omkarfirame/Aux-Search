import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings 
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

## Function To get response from LLAma 2 model




def getLLamaresponse(file,question):
    loader = PyPDFDirectoryLoader(file)
    documents = loader.load()
    embeddings = OllamaEmbeddings()
    print(f"Processed {len(documents)} pdf files")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts=text_splitter.split_documents(documents)
    vector = FAISS.from_documents(texts, embeddings)


    ### LLama2 model
    llm=CTransformers(model='C:\\Users\\omkar\\Downloads\\LLama2-Model\llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {question}""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": {question}})
    print(response["answer"])

    return response["answer"]






st.set_page_config(page_title="Aux-Search",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Aux-Search Information 🤖")

input_text=st.text_input("Upload The pdf")

## creating to more columns for additonal 2 fields

files=st.file_uploader("Choose a file", type=["txt","pdf"])
question = st.text_input("Enter the Question")
    
submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(files,question))