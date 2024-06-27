import streamlit as st
import os
import tempfile

from langchain_community.llms import Ollama
from document_loader import load_documents_into_database
from models import get_list_of_models
from llm import getStreamingChain

EMBEDDING_MODEL = "nomic-embed-text"

st.title("Local LLM with RAG 📚")

if "list_of_models" not in st.session_state:
    st.session_state["list_of_models"] = get_list_of_models()

selected_model = st.sidebar.selectbox(
    "Select a model:", st.session_state["list_of_models"]
)

if st.session_state.get("ollama_model") != selected_model:
    st.session_state["ollama_model"] = selected_model
    st.session_state["llm"] = Ollama(model=selected_model)

# File uploader for multiple files
uploaded_files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True)

if uploaded_files:
    if st.sidebar.button("Index Documents"):
        if "db" not in st.session_state:
            with st.spinner("Creating embeddings and loading documents into Chroma..."):
                # Create a temporary directory to store uploaded files
                with tempfile.TemporaryDirectory() as temp_dir:
                    for file in uploaded_files:
                        file_path = os.path.join(temp_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())

                    # Load documents from the temporary directory
                    st.session_state["db"] = load_documents_into_database(
                        EMBEDDING_MODEL, temp_dir
                    )
            st.info("All set to answer questions!")
else:
    st.warning("Please upload files to load documents into the database.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = getStreamingChain(
            prompt,
            st.session_state.messages,
            st.session_state["llm"],
            st.session_state["db"],
        )
        response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
