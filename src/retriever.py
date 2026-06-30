from langchain.chains import create_history_aware_retriever
from src.prompts import contextualize_q_prompt


def create_retriever(
    llm,
    vector_store,
    k=3,
):

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": k
        }
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_q_prompt,
    )

    return history_aware_retriever