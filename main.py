
import os
from src.loader import load_pdf
from src.splitter import split_documents
from src.embeddings import get_embeddings
from src.vectorstore import create_vectorstore , save_vectorstore, load_vectorstore
from src.retriever import create_retriever
from langchain.chat_models import init_chat_model
from src.rag_chain import create_rag_chain
from langchain_core.messages import HumanMessage, AIMessage
from src.chatbot import get_rag_chain

rag_chain = get_rag_chain()

chat_history = []

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    response = rag_chain.invoke(
        {
            "input": question,
            "chat_history": chat_history,
        }
    )

    print(f"\nBot: {response['answer']}")

    chat_history.append(
        HumanMessage(content=question)
    )

    chat_history.append(
        AIMessage(content=response["answer"])
    )



# response = rag_chain.invoke(
#     {         #Hardcoded question
#         "input": "explain about loan process",

#     }
# )
# print(response["answer"])


# docs = retriever.invoke(
#     "What is Loan Processing?"
# )

# for i, doc in enumerate(docs, start=1):
#     print(f"\nChunk {i}")
#     print(doc.page_content[:200])

# print(f"Retrieved Chunks : {len(docs)}")

# print("-" * 50)

# print(docs[0].page_content)

# print(vector_store.docstore._dict.keys())
# print(vector_store.docstore._dict)
# print(vector_store.index.ntotal)
# print(embeddings)
# print(f"Total Pages : {len(documents)}")
# print(f"Total Chunks : {len(chunks)}")

# print(chunks[0].page_content)

