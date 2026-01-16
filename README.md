# 🤖 AI-Powered RAG Knowledge Base

A professional, production-ready Retrieval-Augmented Generation (RAG) application that allows users to build a searchable knowledge base from PDF and Word documents. This project features a FastAPI backend, a Streamlit frontend, and is fully containerized for cloud deployment.

## 🚀 Live Demo
* **Frontend:** `https://nextwork-rag-frontend.onrender.com/`
* **Backend API:** `https://nextwork-rag-backend.onrender.com`

## ✨ Features
* **Hybrid Document Ingestion:** Supports uploading `.pdf` and `.docx` files or pasting plain text directly into the UI.
* **Smart Text Chunking:** Implements `RecursiveCharacterTextSplitter` to break long documents into 1,000-character chunks with a 200-character overlap to preserve semantic context.
* **Memory-Optimized Embeddings:** Specifically designed for the Render Free Tier (512MB RAM) by utilizing **Voyage AI** for external vector embeddings.
* **High-Performance LLM:** Integrated with **Groq (Llama-3.3-70b)** for lightning-fast, grounded responses.
* **Vector Search:** Uses **ChromaDB** as a persistent vector store to retrieve relevant document snippets.

## 🛠️ Technical Stack
* **Backend:** Python, FastAPI, Uvicorn
* **Frontend:** Streamlit, Requests
* **AI/ML:** Groq API, Voyage AI, LangChain
* **DevOps:** Docker, GitHub Actions (CI/CD), Render

## 🏗️ Architecture
1. **Ingestion:** Documents are parsed and split into overlapping chunks to prevent data loss at boundaries.
2. **Embedding:** Text chunks are converted into vectors via the Voyage AI API.
3. **Storage:** Vectors are stored in a persistent ChromaDB instance.
4. **Retrieval:** User queries are vectorized and compared against the database to find the top relevant chunks.
5. **Generation:** Retrieved context is injected into a prompt for Llama-3 to generate a precise answer.

