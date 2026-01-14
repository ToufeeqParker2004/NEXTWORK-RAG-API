import streamlit as st
import requests
import os


# 1. Setup the page
st.set_page_config(page_title="RAG AI Assistant", page_icon="🤖")
st.title("🤖 My RAG Knowledge Base")

# Define the API endpoints
# Change from http://127.0.0.1:8000 to the Docker service name
API_URL = os.getenv("API_URL", "http://localhost:8000")
QUERY_URL = f"{API_URL}/query"
ADD_URL = f"{API_URL}/add"

# --- SIDEBAR: ADD KNOWLEDGE ---
with st.sidebar:
    st.header("🧠 Expand Knowledge")
    st.write("Paste new technical docs or notes below to add them to ChromaDB.")
    
    new_doc = st.text_area("Content to add:", placeholder="e.g., Docker is a platform for...")
    
    if st.button("Add to Knowledge Base"):
        if new_doc.strip():
            try:
                # Sending the text to your @app.post("/add") endpoint
                response = requests.post(f"{ADD_URL}?text={new_doc}")
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Added! ID: {result.get('id')}")
                else:
                    st.error(f"Failed: {response.status_code}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
        else:
            st.warning("Please enter some text first.")

# --- MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.spinner("Searching and generating..."):
            # Calling your @app.post("/query") endpoint
            response = requests.post(f"{QUERY_URL}?q={prompt}")
            
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer found.")
            else:
                answer = "Error: Could not reach the API."
    except Exception as e:
        answer = f"Connection Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})