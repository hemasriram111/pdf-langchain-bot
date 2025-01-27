from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLLM
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from htmlTemplates import css, bot_template, user_template
import ollama
from pydantic import BaseModel
from typing import Optional, List
from langchain.schema import LLMResult, Generation

# Custom Ollama Embeddings
class OllamaEmbeddings(Embeddings):
    def __init__(self, model_name="llama2"):
        self.model_name = model_name

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings

    def embed_query(self, text):
        response = ollama.embeddings(model=self.model_name, prompt=text)
        return response["embedding"]

# Custom Ollama LLM Wrapper
class OllamaLLM(BaseLLM, BaseModel):
    model_name: str = "llama2"  # Define as a Pydantic field with a default value

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = ollama.generate(model=self.model_name, prompt=prompt)
        return response["response"]

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            response = self._call(prompt, stop)
            generations.append([Generation(text=response)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "ollama"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model_name="llama2")  # Use Ollama embeddings
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = OllamaLLM(model_name="llama2")  # Use the custom Ollama LLM wrapper

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process a PDF before asking a question.")
        return
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                if not pdf_docs:
                    st.warning("Please upload a PDF before processing.")
                    return

                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

if __name__ == '__main__':
    main()
