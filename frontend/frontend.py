import streamlit as st
import requests
import os

# 1. Setup the page
st.set_page_config(page_title="RAG AI Assistant", page_icon="🤖")
st.title("🤖 My RAG Knowledge Base")

# Define the API endpoints
# Defaults to localhost if API_URL environment variable isn't set in Render
API_URL = os.getenv("API_URL", "http://localhost:8000")
QUERY_URL = f"{API_URL}/query"
ADD_URL = f"{API_URL}/add"

# --- SIDEBAR: UPLOAD DOCUMENTS ---
with st.sidebar:
    st.header("🧠 Expand Knowledge")
    st.write("Upload PDF or Word documents to add them to your knowledge base.")
    
    # Updated to accept files instead of just text
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
    
    if st.button("Index Document"):
        if uploaded_file:
            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Prepare the file for the multipart/form-data request
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    
                    # Sending the file to your updated @app.post("/add") endpoint
                    response = requests.post(ADD_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Success! {result.get('message')}")
                else:
                    st.error(f"Failed to index: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
        else:
            st.warning("Please select a file first.")

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
            response = requests.post(QUERY_URL, params={"q": prompt})
            
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer found.")
            else:
                answer = f"Error: {response.status_code} - Could not reach the API."
    except Exception as e:
        answer = f"Connection Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})