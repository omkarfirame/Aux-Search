from langchain.chains.question_answering import load_qa_chain
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from AuxSearch.constants import *

class ChainLoader:
    def __init__(self):
        self.model_path = MODEL_PATH

    def get_chains(self):
        """
        parameter: prompt_template
        return: chain
        """

        prompt_template = """
            Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
            provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
            Context:\n {context}?\n
            Question: \n{question}\n

            Answer:
        """
        model = CTransformers(model = 'C:/Users/omkar/Downloads/LLama2-Model/llama-2-7b-chat.ggmlv3.q8_0.bin',model_type='llama')
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
