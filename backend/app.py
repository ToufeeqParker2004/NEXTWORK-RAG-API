import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# 1. Initialize FastAPI app
app = FastAPI(title="RAG API with Groq")

# 2. Initialize Groq Client
# Ensure you have set GROQ_API_KEY in your environment or GitHub Secrets
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 3. Setup ChromaDB (Your Knowledge Base)
# This persists your data even if the container restarts
client = chromadb.PersistentClient(path="./chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(
    name="my_knowledge_base",
    embedding_function=sentence_transformer_ef
)

# 4. Data Models for API
class QueryRequest(BaseModel):
    question: str

# 5. API Endpoints

@app.get("/")
def home():
    return {"message": "RAG API is running with Groq!"}

@app.post("/add")
def add_to_knowledge(text: str):
    """Adds a new document to the vector database."""
    try:
        # Generate a simple unique ID based on collection count
        doc_id = str(collection.count() + 1)
        collection.add(
            documents=[text],
            ids=[doc_id]
        )
        return {"message": "Document added successfully", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_knowledge(q: str):
    """Retrieves context from ChromaDB and generates an answer using Groq."""
    try:
        # STEP 1: Retrieval - Find the most relevant context in ChromaDB
        results = collection.query(
            query_texts=[q],
            n_results=2
        )
        
        # Flatten the retrieved documents into a single string
        context = " ".join(results['documents'][0]) if results['documents'] else "No context found."

        # STEP 2: Augmentation & Generation - Send context + question to Groq
        prompt = f"Answer the following question using ONLY the provided context.\n\nContext: {context}\n\nQuestion: {q}"
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided documents."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile", # High-performance model
        )

        return {
            "answer": chat_completion.choices[0].message.content,
            "context_used": context
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq/Chroma Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)