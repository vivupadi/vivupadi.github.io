import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import os
import re
import pickle
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import InferenceClient

# Constants
INDEX_PATH = "faiss_index.pkl"
#MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  #To be used to vectorstore (pdf to vectorstore for first time)
#QA_MODEL = "deepset/roberta-base-squad2"

import requests

#API_URL = "https://sqix2hh8u16b2j8i.us-east-1.aws.endpoints.huggingface.cloud"  # Replace this
#HF_TOKEN = 'Hugging_Face_Token'       # Your HF access token

#headers = {
#    "Authorization": f"Bearer {HF_TOKEN}",
#    "Content-Type": "application/json"
#}

# Streamlit app configuration
st.set_page_config(page_title="CV QA System", layout="wide")
st.title("Chatbot to answer Quick Questions")
st.markdown("Ask questions about Vivek Padayattil")

def preprocess_text(text):
    """Clean PDF text artifacts"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'•|\uf0b7', '-', text)
    text = re.sub(r'\s[BH]\s', ' ', text)
    return text.strip()

@st.cache_resource
def load_or_process_documents():
    """Cache processed documents to avoid reprocessing"""
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, 'rb') as f:
            return pickle.load(f)
    
    loader = PyPDFLoader("Vivek_Padayattil_CV_2.pdf")
    documents = loader.load()
    #print(documents)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=2,
        #separators=["\n\n", "\n", "(?<=\. )", " "]
        separators=["\n\n", "\n", " o "," { ", "(?<=\ )"]
    )
    
    texts = [preprocess_text(doc.page_content) for doc in documents]
    chunks = text_splitter.create_documents(texts)
    #print(chunks)
    
    #embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)  #only required first time
    #vectorstore = FAISS.from_documents(chunks, embeddings)   # only required first time
    #print(vectorstore)
    with open(INDEX_PATH, 'wb') as f:
        pickle.dump(vectorstore, f)
    
    return vectorstore

@st.cache_resource
def initialize_qa_model():
    """Lightweight local QA model"""
    #client = InferenceClient(model="tiiuae/falcon-rw-1b", token = os.getenv("Hugging_Face_Token"))
    return InferenceClient(model="deepset/roberta-base-squad2", token = os.getenv("Hugging_Face_Token"))
    return pipeline(
        "text2text-generation", ##new addition
        model="google/flan-t5-base",
        device=-1  # CPU
    )

def retrieve_context(query, vectorstore, k=3):
    """Enhanced retrieval with score thresholding"""
    docs = vectorstore.similarity_search_with_score(query, k=k)
    #print(docs)
    #lowest_doc = min(docs, key= lambda doc: doc[1])
    #return [lowest_doc[0].page_content]
    return [doc[0].page_content for doc in docs if doc[1] < 1.8]

# Initialize components
vectorstore = load_or_process_documents()
#qa_pipeline = initialize_qa_model()
#qa_pipeline = InferenceClient(model="deepset/roberta-base-squad2", token = os.getenv('Hugging_Face_Token'))
qa_pipeline = InferenceClient(
    provider="novita",
    api_key=os.getenv('Hugging_Face_Token'),
)
#qa_pipeline= pipeline(
#        "text2text-generation", ##new addition
#        model="google/flan-t5-base",
#        device=-1  # CPU
#   )

# Sidebar for additional controls
#with st.sidebar:
    #st.header("Settings")
    #k_value = st.slider("Number of context chunks", 1, 5, 3)
    #confidence_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.25)

# Main chat interface
user_question = st.chat_input("Ask a question about Vivek...")

if user_question:
    with st.spinner("Searching for answers..."):
        # Display user question
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Retrieve context
        contexts = retrieve_context(user_question, vectorstore, k=3)
        #print(contexts)
        
        # Get best answer
        best_answer = ""
        for context in contexts:
            #prompt = f"Answer the question based on the context below. Context: {context} \nQuestions: {user_question}\nAnswer:"
            result = qa_pipeline.question_answering(question=user_question, context=context)  #for explorativeQA
            #result = qa_pipeline(prompt, max_length=512)[0]['generated_text']
            #result = qa_pipeline.text_generation(prompt, max_new_tokens=100)
            #result = requests.post(API_URL, headers=headers, json={"inputs": prompt})
            #print(result.json()[0]["generated_text"])
            print(result)
            #best_answer = result["answer"]
            if result["score"] > 0.05:
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