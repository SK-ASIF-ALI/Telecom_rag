import shutil
import os
from Telecom_data_loader import *

# Delete old vectorstore
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
    print("✅ Deleted old chroma_db")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

if __name__ == "__main__":
    obj = telecom()
    
    # Load documents
    docs = obj.load_all_documents()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    
    # Create vectorstore with consistent embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print("✅ Vector store ready!")
    
    # Build RAG chain
    rag_chain = obj.build_rag(vectorstore)
    print("✅ RAG system ready! Type 'quit' to exit.\n")
    
    # Interactive loop
    while True:
        query = input("You: ")
        if query.lower() == 'quit':
            break
        
        response = rag_chain(query)
        print(f"\nAssistant: {response}\n")