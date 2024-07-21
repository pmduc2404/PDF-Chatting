import streamlit as st
from PyPDF2 import PdfReader
from database_access import *
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter, SentenceTransformersTokenTextSplitter, RecursiveCharacterTextSplitter, TextSplitter, TokenTextSplitter
import time
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
from langchain_huggingface import HuggingFacePipeline
import os

if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    st.session_state.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    st.session_state.text2text_pipeline = hf_pipeline('text2text-generation', model=st.session_state.model, tokenizer=st.session_state.tokenizer)
    st.session_state.llm = HuggingFacePipeline(pipeline=st.session_state.text2text_pipeline)

if 'text_splitter' not in st.session_state:
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
db_name = "chroma_pdf"
persistent_directory = os.path.join("./database", db_name)

# readding pdf with metadata
def read_pdf(files):
    documents = []
    for file in files:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        
        #
        doc = Document(
            page_content=text,
            metadata={"source": file.name}  # 
        )
        
        documents.append(doc)
    return documents

def create_chains(retriever, llm):
    # Contextualize question
    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    qa_system_prompt = """You are an assistant for question-answering tasks. \
                            Use the following pieces of retrieved context to answer the question. \
                            If you don't know the answer, just say that you don't know. \
                            Use three sentences maximum and keep the answer concise.\{context}"""
                            
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

def main():
    st.set_page_config(page_title="AI Chat", page_icon="🤖")

    # Initialize session state if not exists
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
        st.session_state.chat_history = []

    st.header("Chat with the PDF")
    
    # Sidebar for file upload
    with st.sidebar:
        pdfs = st.file_uploader("Upload your file here", type=['pdf'], accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Loading"):
                documents = read_pdf(pdfs)
    
                if not os.path.exists(persistent_directory):
                    print("Persistent directory does not exist. Initializing vector store...")
                    st.session_state.retriever = create_vector_store(persistent_directory, documents, st.session_state.text_splitter)
                else:
                    st.session_state.retriever = add_vector(persistent_directory, documents, st.session_state.text_splitter)
                    
                st.session_state.conversation_chain = create_chains(st.session_state.retriever, st.session_state.llm)
                st.write("Upload and initialization complete.")
    
    # Create a form to hold the question input and button
    with st.form(key='question_form'):
        user_question = st.text_input(label="Question here", placeholder="Enter your question...")
        submit_button = st.form_submit_button(label="Ask")

        if submit_button:
            if st.session_state.conversation_chain:
                with st.spinner("Processing"):
                    result = st.session_state.conversation_chain.invoke({"input": user_question, "chat_history": st.session_state.chat_history})
                    st.session_state.chat_history.append({"role": "user", "content": user_question})
                    st.session_state.chat_history.append({"role": "assistant", "content": result['answer']})
                    st.write(result['answer'])
            else:
                st.write("Please upload and initialize the PDF first.")
    
if __name__ == "__main__":
    main()