
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import RetrievalQA
#from langchain.document_loaders import JSONLoader
from langchain_community.document_loaders import JSONLoader
#from langchain.vectorstores import FAISS
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
#from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load your CV as a JSON document
loader = JSONLoader(file_path="C:\\Users\\Vivupadi\\Desktop\\Portfolio\\Vivek_Resume.json", jq_schema=".")
documents = loader.load()

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# Set up Hugging Face model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Create RAG-based chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('message', '')
    response = qa_chain.run(question)
    return jsonify({"reply": response})

if __name__ == '__main__':
    app.run(debug=True)