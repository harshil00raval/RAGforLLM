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

    # Encode documents and normalize for cosine similarity
    embeddings = embedding_model.encode(docs)

    # Normalize embeddings for cosine similarity
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Use IndexFlatIP for inner product (cosine similarity when vectors are normalized)
    dimension = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_embeddings.astype(np.float32))

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

def normalize(vectors):
    """Normalize vectors to unit length for cosine similarity."""
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

@app.get("/query/")
def query_faiss(query: str):
    global index
    if index is None:
        raise HTTPException(status_code=400, detail="No documents indexed")

    k = min(5, len(docs))  # Number of closest matches to fetch (limit to available docs)
    threshold = 0.2  # Cosine similarity threshold (between 0 and 1)

    # Encode and normalize query vector
    query_embedding = embedding_model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = query_embedding.astype(np.float32)

    # With IndexFlatIP and normalized vectors, scores will be cosine similarities
    scores, indices = index.search(query_embedding, k)

    # For inner product with normalized vectors, similarity is directly the score
    # (no need for distance-to-similarity conversion)
    similarity_scores = scores[0]

    filtered_results = [
        {
            "file": file_names[idx],
            "content": docs[idx][:200] + "..." if len(docs[idx]) > 200 else docs[idx],  # Truncate for display
            "score": float(round(score, 3))
        }
        for idx, score in zip(indices[0], similarity_scores)
        if idx != -1 and score > threshold
    ]

    if not filtered_results:
        return {
            "message": "No relevant match found above threshold.",
            "debug_info": {
                "top_scores": [float(s) for s in similarity_scores],
                "threshold": threshold,
                "num_docs_indexed": len(docs)
            }
        }

    return {"matches": filtered_results, "total_results": len(filtered_results)}

@app.get("/list_files/")
def list_files():
    if not file_names:
        return {"message": "No files are currently encoded in FAISS."}

    file_list = [{"file_name": file, "index": idx} for idx, file in enumerate(file_names)]

    return {"encoded_files": file_list}