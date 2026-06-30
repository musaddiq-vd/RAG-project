from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompts import qa_prompt


def create_rag_chain(
    llm,
    retriever,
):

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
    )

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain,
    )

    return rag_chain