import re
import os
from dotenv import load_dotenv

import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

load_dotenv()

def preprocess_text(text):
    """Clean PDF text artifacts"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'•|\uf0b7', '-', text)
    text = re.sub(r'\s[BH]\s', ' ', text)
    return text.strip()


def create_datastore(file):
    loader = PyPDFLoader(file)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=2,
        #separators=["\n\n", "\n", "(?<=\. )", " "]
        separators=["\n\n", "\n", " o "," { ", "(?<=\ )"]
    )
    
    texts = [preprocess_text(doc.page_content) for doc in documents]
    chunks = text_splitter.create_documents(texts)
    #print(chunks)
    
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)  #only required first time
    vectorstore = FAISS.from_documents(chunks, embeddings)   # only required first time
    #print(vectorstore)
    with open(INDEX_PATH, 'wb') as f:
        pickle.dump(vectorstore, f)
    
    return vectorstore

INDEX_PATH = "faiss_index.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

file = "Vivek_Padayattil_CV_2.pdf"

vectorstore = create_datastore(file)

if vectorstore:
    print('vectorstore created')
