from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load saved index and chunk data
index = faiss.read_index("faiss_index.bin")

with open("chunks_data.pkl", "rb") as f:
    data = pickle.load(f)
    all_chunks = data["chunks"]
    chunk_sources = data["sources"]

# Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, top_k=3):
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    print(f"\nQuery: {query}")
    print("-" * 50)
    for rank, idx in enumerate(indices[0]):
        print(f"Rank {rank+1} | Source: {chunk_sources[idx]} | Distance: {distances[0][rank]:.4f}")
        print(all_chunks[idx][:200], "...\n")

# Try a few test questions
retrieve("When did Beyonce start becoming popular?")
retrieve("What is solar energy?")
retrieve("Who wrote To Kill a Mockingbird?")