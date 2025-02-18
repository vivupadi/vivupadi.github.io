
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import RetrievalQA
#from langchain.document_loaders import JSONLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.llms import HuggingFacePipeline
from transformers import pipeline

import requests

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('api_key')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

#Load the model
API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
API_KEY = os.getenv('api_key')
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Load your CV as a JSON document
#loader = PyPDFLoader(file_path="C:\\Users\\Vivupadi\\Desktop\\Portfolio\\Vivek Padayattil_CV_2024.pdf")
loader = PyPDFLoader(file_path="Vivek Padayattil_CV_2024.pdf")
documents = loader.load()

# Extract text from documents
text_chunks = [doc.page_content for doc in documents]

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(text_chunks, embeddings)

# Set up Hugging Face model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
#llm = HuggingFacePipeline(pipeline=qa_pipeline)


""""
def query_deepseek(question, context):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer YOUR_API_KEY", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "system", "content": "Answer based on context"},
                     {"role": "user", "content": f"Question: {question}\nContext: {context}"}]
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No answer found.")
"""
# Retrieval function
def retrieve_relevant_text(query):
    """Retrieve the most relevant CV section from FAISS."""
    results = vectorstore.similarity_search(query, k=2)
    return results[0].page_content if results else "No relevant information found."

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('message', '')
     
    # Print received question
    print(f"Question received: {question}")
    
    # Retrieve relevant context
    context = retrieve_relevant_text(question)
    print(f"Context for question '{question}': {context}")

    # Get answer from the QA model
    response = qa_pipeline({"question": question, "context": context})
    
    # Check the response from the QA model
    print(f"QA response: {response}")

    return jsonify({"reply": response["answer"]})

if __name__ == '__main__': 
    port = int(os.getenv('PORT', 10000))  # Render provides PORT only required for render
    app.run(debug=True)