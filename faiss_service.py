from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index Setup
index = None
docs = []
file_names = []
file_to_index = {}  # Maps file names to FAISS indices

class DirectoryRequest(BaseModel):
    directory: str

def create_faiss_index():
    global index, docs, file_names
    if not docs:
        index = None
        return
    embeddings = embedding_model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

@app.post("/add_files/")
def add_files(req: DirectoryRequest):
    global docs, file_names, file_to_index, index
    directory = req.directory

    if not os.path.exists(directory):
        raise HTTPException(status_code=400, detail="Directory not found")

    supported_extensions = {".txt", ".java", ".py", ".js", ".md"}

    # Read and encode new files
    for file in os.listdir(directory):
        if file.endswith(tuple(supported_extensions)):
            file_path = os.path.join(directory, file)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                docs.append(content)
                file_names.append(file)
                file_to_index[file] = len(file_names) - 1  # Track file index

    create_faiss_index()
    return {"message": f"Added {len(docs)} documents to FAISS"}

class FileRequest(BaseModel):
    file_name: str

@app.post("/remove_file/")
def remove_file(req: FileRequest):
    global docs, file_names, file_to_index, index

    if req.file_name not in file_names:
        raise HTTPException(status_code=404, detail="File not found in FAISS")

    file_idx = file_names.index(req.file_name)

    # Remove from lists
    del docs[file_idx]
    del file_names[file_idx]
    del file_to_index[req.file_name]

    # Rebuild FAISS Index
    create_faiss_index()

    return {"message": f"Removed {req.file_name} from FAISS"}

@app.get("/query/")
def query_faiss(query: str):
    if not index:
        raise HTTPException(status_code=400, detail="No documents indexed")

    query_embedding = embedding_model.encode([query])
    _, indices = index.search(np.array(query_embedding), 1)
    return {"file": file_names[indices[0][0]], "content": docs[indices[0][0]]}

@app.get("/list_files/")
def list_files():
    if not file_names:
        return {"message": "No files are currently encoded in FAISS."}

    file_list = [{"file_name": file, "index": idx} for idx, file in enumerate(file_names)]

    return {"encoded_files": file_list}