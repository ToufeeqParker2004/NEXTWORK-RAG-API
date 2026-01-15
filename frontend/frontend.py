import streamlit as st
import requests
import os

# 1. Setup the page
st.set_page_config(page_title="RAG AI Assistant", page_icon="🤖")
st.title("🤖 My RAG Knowledge Base")

# Define API URLs
API_URL = os.getenv("API_URL", "http://localhost:8000")
QUERY_URL = f"{API_URL}/query"
ADD_URL = f"{API_URL}/add"

# --- SIDEBAR: ADD KNOWLEDGE ---
with st.sidebar:
    st.header("🧠 Expand Knowledge")
    
    # Input Method 1: File Upload
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx"])
    
    # Input Method 2: Text Area
    new_doc_text = st.text_area("OR Paste text below:", placeholder="Enter technical notes...")
    
    if st.button("Add to Knowledge Base"):
        try:
            with st.spinner("Indexing content..."):
                if uploaded_file:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(ADD_URL, files=files)
                elif new_doc_text.strip():
                    # Send text as Form Data to match backend expectations
                    response = requests.post(ADD_URL, data={"text": new_doc_text})
                else:
                    st.warning("Please provide a file or text.")
                    st.stop()

                if response.status_code == 200:
                    st.success(response.json().get("message", "Success!"))
                else:
                    st.error(f"Error: {response.json().get('detail')}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# --- MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.spinner("Searching knowledge base..."):
            response = requests.post(QUERY_URL, params={"q": prompt})
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer generated.")
            else:
                answer = "Error: Backend is currently unavailable."
    except Exception as e:
        answer = f"Connection Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})