
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import RetrievalQA
#from langchain.document_loaders import JSONLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.llms import HuggingFacePipeline
from transformers import pipeline

import requests

from dotenv import load_dotenv
import os
import re
import pickle

load_dotenv()
API_KEY = os.getenv('api_key')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

#Load the model
API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
API_KEY = os.getenv('api_key')
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Constants
INDEX_PATH = "faiss_index.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "deepset/minilm-uncased-squad2"  # Lighter than roberta-base


def preprocess_text(text):
    """Clean PDF text artifacts"""
    text = re.sub(r'\s+', ' ', text)  # Remove excessive whitespace
    text = re.sub(r'•|\uf0b7', '-', text)  # Normalize bullet points
    return text.strip()

def load_or_process_documents():
    """Cache processed documents to avoid reprocessing"""
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, 'rb') as f:
            return pickle.load(f)
    
    loader = PyPDFLoader("static/Vivek Padayattil_CV_2024.pdf")
    documents = loader.load()
    
    # Better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        separators=["\n\n", "\n", "(?<=\. )", " "]
    )
    
    texts = [preprocess_text(doc.page_content) for doc in documents]
    chunks = text_splitter.create_documents(texts)
    
    # Generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Cache results
    with open(INDEX_PATH, 'wb') as f:
        pickle.dump(vectorstore, f)
    
    return vectorstore

# --- QA System ---
def initialize_qa_model():
    """Lightweight local QA model"""
    return pipeline(
        "question-answering", 
        model=QA_MODEL,
        tokenizer=QA_MODEL,
        device=-1  # CPU
    )


# --- Retrieval Optimization ---
def retrieve_context(query, vectorstore, k=3):
    """Enhanced retrieval with score thresholding"""
    docs = vectorstore.similarity_search_with_score(query, k=k)
    return [doc[0].page_content for doc in docs if doc[1] > 0.7]  # Score threshold

# --- API Endpoint ---
vectorstore = load_or_process_documents()
qa_pipeline = initialize_qa_model()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('message', '').strip()
        
        if not question:
            return jsonify({"reply": "Please enter a valid question"}), 400
        
        # Retrieve context
        contexts = retrieve_context(question, vectorstore)
        if not contexts:
            return jsonify({"reply": "No relevant information found in my CV"})
        
        # Get best answer across contexts
        best_answer = ""
        for context in contexts:
            result = qa_pipeline(question=question, context=context)
            if result["score"] > 0.5:  # Confidence threshold
                best_answer = result["answer"]
                break
        
        return jsonify({"reply": best_answer or "I couldn't find a definitive answer in my CV"})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"reply": "An error occurred while processing your question"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)