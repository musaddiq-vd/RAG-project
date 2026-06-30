from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question, "
            "rewrite the question into a standalone question. "
            "Do not answer it.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Answer the user's question using only the provided context.

If the answer is not present in the context, say:
'I don't know based on the provided document.'

Context:
{context}
""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)