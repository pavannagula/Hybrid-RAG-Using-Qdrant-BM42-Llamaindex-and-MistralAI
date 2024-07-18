import requests
import streamlit as st 
from generation import prompt_template_generation, create_query_engine

# Title
st.set_page_config(page_title="Hybrid RAG with Qdrant BM42 & Mistral", layout="wide")
st.title("Hybrid RAG with Qdrant BM42 & Mistral 8x7B")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input for query and filename
query = st.chat_input("Ask me anything about different RAG frameworks!")
filename = st.text_input("Do you want to pick any specific filename:")

if query and filename:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        query_str = st.session_state.messages[-1]["content"]
        prompt_gen = prompt_template_generation()
        prompt = prompt_gen.prompt_generation(query=query_str, filename=filename)
        response = create_query_engine(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})