from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import sqlite3
import os
import logging
from dotenv import load_dotenv

load_dotenv()


class telecom():
    def __init__(self):
        logging.info("Initializing the telecom class")
        
    def build_rag(self, vectorstore):
        # Groq model (LLM)
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

        # Prompt template
        system_prompt = (
            "You are a helpful telecom support assistant. "
            "Use retrieved context to answer. When unsure, say \"I don't have enough information\"."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {input}")
        ])

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Simple RAG chain using pipe operator
        rag_chain = prompt | llm | StrOutputParser()
        
        def answer_question(question):
            # context_docs = retriever.get_relevant_documents(question)
            context_docs = retriever.invoke(question)
            context = "\n".join([doc.page_content for doc in context_docs])
            return rag_chain.invoke({"context": context, "input": question})
        
        return answer_question

    def load_all_documents(self,
        pdf_path: str = "data/telecom_guide.pdf",
        excel_path: str = "data/faq.csv",
        db_path: str = "data/tickets.db"
    ):
        docs = []

        # 1. PDF
        pdf_loader = PyPDFLoader(pdf_path)
        pdf_docs = pdf_loader.load()
        docs.extend(pdf_docs)
        print(f"✅ Loaded {len(pdf_docs)} pages from PDF")

        # 2. Excel (FAQ)
        df = pd.read_csv(excel_path)
        df_loader = DataFrameLoader(df, page_content_column="answer")
        excel_docs = df_loader.load()
        docs.extend(excel_docs)
        print(f"✅ Loaded {len(excel_docs)} FAQs from Excel")

        # 3. SQLite (Tickets)
        conn = sqlite3.connect(db_path)
        df_tickets = pd.read_sql_query("SELECT * FROM tickets", conn)
        conn.close()

        for _, row in df_tickets.iterrows():
            content = " | ".join([f"{col}: {val}" for col, val in row.items()])
            docs.append(Document(page_content=content, metadata={"source": "tickets.db"}))
        print(f"✅ Loaded {len(df_tickets)} tickets from DB")

        return docs