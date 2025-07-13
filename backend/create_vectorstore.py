import re
import os
from dotenv import load_dotenv

import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from sentence_transformers import SentenceTransformer


load_dotenv()

def get_embedding_model(model_name):
    # Use SentenceTransformer directly to ensure tokenizer properties can be set
    embed_model = SentenceTransformer(model_name)
    if embed_model.tokenizer.pad_token is None:
        embed_model.tokenizer.pad_token = embed_model.tokenizer.eos_token or '[PAD]'
    return embed_model

def preprocess_text(text):
    """Clean PDF text artifacts"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'•|\uf0b7', '-', text)
    text = re.sub(r'\s[BH]\s', ' ', text)
    return text.strip()


def create_datastore(file, MODEL_NAME):
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
    
    #Use your embedding model with pad_token fixed
    #embed_model = get_embedding_model(MODEL_NAME)
    
    #embeddings = HuggingFaceEmbeddings(model=embed_model)

    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)  #only required first time
    vectorstore = FAISS.from_documents(chunks, embeddings)   # only required first time
    vectorstore.save_local('faiss_index')
    #print(vectorstore)
    #with open(INDEX_PATH, 'wb') as f:
    #    pickle.dump(vectorstore, f)
    
    return vectorstore

INDEX_PATH = "faiss_index.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

file = "Vivek_Padayattil_CV_2.pdf"

vectorstore = create_datastore(file, MODEL_NAME)

if vectorstore:
    print('vectorstore created')
