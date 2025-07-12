import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from huggingface_hub import InferenceClient
from langchain.schema import Document
import os
import re
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv

load_dotenv()


# Constants
INDEX_PATH = "faiss_index.pkl"

import requests

# Streamlit app configuration
st.set_page_config(page_title="CV QA System", layout="wide")
st.title("Chatbot to answer Quick Questions")
st.markdown("Ask questions about Vivek Padayattil")

@st.cache_resource
def load_or_process_documents():
    """Cache processed documents to avoid reprocessing"""
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, 'rb') as f:
            return pickle.load(f)
    
    print("Vectorstore returned")
    return vectorstore

@st.cache_resource
def initialize_qa_model():
    """Lightweight local QA model"""
    return InferenceClient(model="deepset/roberta-base-squad2", token = os.getenv("Hugging_Face_Token"))

def retrieve_context(query, vectorstore, k=1):
    """Enhanced retrieval with score thresholding"""
    docs = vectorstore.similarity_search_with_score(query, k=5)
    return [doc[0].page_content for doc in docs if doc[1] < 1.8]

# Initialize components
vectorstore = load_or_process_documents()
qa_pipeline = initialize_qa_model()

# Main chat interface
user_question = st.chat_input("Ask a question about Vivek...")

if user_question:
    with st.spinner("Searching for answers..."):
        # Display user question
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Retrieve context
        contexts = retrieve_context(user_question, vectorstore, k=5)
        #print(contexts)
        
        # Get best answer
        #best_answer = ""
        for context in contexts:
            result = qa_pipeline.question_answering(question=user_question, context=context)  #for explorativeQA
            best_answer = result["answer"]
            if result["score"] > 0.1:
                best_answer = result["answer"]
                break
        

        # Display response
        with st.chat_message("assistant"):
            if best_answer:
                st.success(best_answer)
                with st.expander("See relevant context"):
                    st.write(context)
            elif contexts:
                st.warning("I found some relevant information but wasn't confident enough to answer.")
                with st.expander("See potential context"):
                    st.write("\n\n---\n\n".join(contexts))
            else:
                st.error("No relevant information found in the CV.")