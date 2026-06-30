from langchain_community.vectorstores import FAISS


def create_vectorstore(chunks, embeddings):

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


def save_vectorstore(vector_store, path="data/db"):

    vector_store.save_local(path)


def load_vectorstore(embeddings, path="data/db"):

    vector_store = FAISS.load_local(
        folder_path=path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store

# chunks → Splitter se aaye.
# embeddings → Titan model.
# FAISS.from_documents() → Embeddings create + FAISS index create.