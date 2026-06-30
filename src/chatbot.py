import os

from src.loader import load_pdf
from src.splitter import split_documents
from src.embeddings import get_embeddings
from src.vectorstore import (
    create_vectorstore,
    save_vectorstore,
    load_vectorstore,
)
from src.retriever import create_retriever
from src.rag_chain import create_rag_chain

from langchain.chat_models import init_chat_model

def get_rag_chain():
    embeddings = get_embeddings()

    if os.path.exists("data/db"):

        print("Loading existing vector store...")

        vector_store = load_vectorstore(
            embeddings=embeddings,
        )

    else:

        print("Creating new vector store...")

        documents = load_pdf("data/sample.pdf")

        chunks = split_documents(
        documents=documents,
        chunk_size=500,
        chunk_overlap=100,
        )

        vector_store = create_vectorstore(
            chunks=chunks,
            embeddings=embeddings,
        )

        save_vectorstore(
            vector_store=vector_store,
        )


    #Initialize LLM
    llm = init_chat_model(
        "amazon.nova-pro-v1:0",
        model_provider="bedrock_converse",
    )

    #Create Retriever
    retriever = create_retriever(
        llm=llm,
        vector_store=vector_store,
        k=3,
    )

    #Create RAG Chain
    rag_chain = create_rag_chain(
        llm=llm,
        retriever=retriever,
    )

    return rag_chain