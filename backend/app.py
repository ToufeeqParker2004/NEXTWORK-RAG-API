import os
from fastapi import FastAPI
import chromadb
from ollama import Client  # 1. Import the Client class

app = FastAPI()

# 2. Get the Ollama host from an environment variable (set in your k8s manifest)
# Default to localhost so it still works if you run it normally
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_client = Client(host=OLLAMA_HOST)

chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")

@app.post("/query")
def query(q: str):
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"] else ""

    # 3. Use the client instance instead of the global 'ollama' module
    answer = ollama_client.generate(
        model="tinyllama",
        prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:"
    )

    return {"answer": answer["response"]}

@app.post("/add")
def add_knowledge(text: str):
    """Add new content to the knowledge base dynamically."""
    try:
        # Generate a unique ID for this document
        import uuid
        doc_id = str(uuid.uuid4())
        
        # Add the text to Chroma collection
        collection.add(documents=[text], ids=[doc_id])
        
        return {
            "status": "success",
            "message": "Content added to knowledge base",
            "id": doc_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
