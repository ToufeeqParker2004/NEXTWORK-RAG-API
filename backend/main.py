import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF for PDFs
import docx  # for Word files
from io import BytesIO
from typing import Optional

# 1. Initialize FastAPI app
app = FastAPI(title="RAG API with Groq and Voyage AI")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Initialize Clients
# Make sure VOYAGE_API_KEY and GROQ_API_KEY are in Render Environment Variables
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# API-based embedding saves 300MB+ of RAM compared to local models
voyage_ef = embedding_functions.VoyageAIEmbeddingFunction(
    api_key=os.environ.get("VOYAGE_API_KEY"),
    model_name="voyage-2"
)

# 3. Setup ChromaDB and Text Splitter
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="my_knowledge_base",
    embedding_function=voyage_ef
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

# 4. API Endpoints

@app.get("/")
def home():
    return {"message": "RAG API is live with Hybrid Uploads!"}

@app.post("/add")
async def add_to_knowledge(
    file: Optional[UploadFile] = File(None), 
    text: Optional[str] = Form(None)
):
    """Accepts either an uploaded file OR a text string."""
    text_content = ""
    source_name = ""

    try:
        # Priority 1: Check for Text Paste
        if text and text.strip():
            text_content = text
            source_name = f"manual_entry_{collection.count() + 1}"

        # Priority 2: Check for File Upload
        elif file:
            source_name = file.filename
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            if file_extension == ".pdf":
                pdf_bytes = await file.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                text_content = "\n".join([page.get_text() for page in doc])
            elif file_extension == ".docx":
                docx_bytes = await file.read()
                doc = docx.Document(BytesIO(docx_bytes))
                text_content = "\n".join([para.text for para in doc.paragraphs])
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format.")
        
        else:
            raise HTTPException(status_code=400, detail="No content provided.")

        # Chunk and Index
        chunks = text_splitter.split_text(text_content)
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                ids=[f"{source_name}_chunk_{i}"],
                metadatas=[{"source": source_name}]
            )
            
        return {"message": f"Successfully indexed {len(chunks)} chunks from {source_name}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_knowledge(q: str):
    try:
        results = collection.query(query_texts=[q], n_results=3)
        context = " ".join(results['documents'][0]) if results['documents'] else "No context found."

        prompt = f"Answer the following question using ONLY the provided context.\n\nContext: {context}\n\nQuestion: {q}"
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
        )

        return {
            "answer": chat_completion.choices[0].message.content,
            "context_used": context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)