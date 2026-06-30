import streamlit as st

# RAG Chain
from src.chatbot import get_rag_chain
from langchain_core.messages import HumanMessage, AIMessage


# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Loan Assistant",
    page_icon="🤖",
    layout="wide",
)


# ==========================================================
# Title
# ==========================================================
st.title("🤖 Loan Assistant")
st.caption("Ask anything about your Loan Documents")


# ==========================================================
# Load RAG Chain only once
# Streamlit cache keeps the chain in memory.
# It avoids loading FAISS and LLM on every refresh.
# ==========================================================
@st.cache_resource
def load_chain():
    return get_rag_chain()


rag_chain = load_chain()


# ==========================================================
# Session State
# Stores complete conversation for current browser session.
# ==========================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==========================================================
# Sidebar
# ==========================================================
with st.sidebar:

    st.header("Settings")

    if st.button("🗑 Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()


# ==========================================================
# Display Previous Chat Messages
# ==========================================================
for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================================
# Chat Input
# ==========================================================
question = st.chat_input("Ask your question...")


# ==========================================================
# If user asks a question
# ==========================================================
if question:

    # -----------------------------
    # Show User Message
    # -----------------------------
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # -----------------------------
    # Generate Bot Response
    # -----------------------------
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = rag_chain.invoke(
                {
                    "input": question,
                    "chat_history": [
                        HumanMessage(content=msg["content"])
                        if msg["role"] == "user"
                        else AIMessage(content=msg["content"])
                        for msg in st.session_state.chat_history
                    ],
                }
            )

            answer = response["answer"]

            st.markdown(answer)

    # -----------------------------
    # Save Bot Response
    # -----------------------------
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )