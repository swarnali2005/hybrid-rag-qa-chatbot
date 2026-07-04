from sentence_transformers import SentenceTransformer
import faiss
import pickle

# --- Load FAISS index + chunk data ---
faiss_index = faiss.read_index("faiss_index.bin")
with open("chunks_data.pkl", "rb") as f:
    faiss_data = pickle.load(f)
    all_chunks = faiss_data["chunks"]
    chunk_sources = faiss_data["sources"]

# --- Load BM25 index ---
with open("bm25_index.pkl", "rb") as f:
    bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def dense_retrieve(query, top_k=10):
    query_vector = embed_model.encode([query]).astype("float32")
    distances, indices = faiss_index.search(query_vector, top_k)
    return list(indices[0])  # ranked list of chunk indices, best first

def sparse_retrieve(query, top_k=10):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return ranked_indices  # ranked list of chunk indices, best first

def reciprocal_rank_fusion(dense_ranks, sparse_ranks, k=60):
    """
    RRF formula: score(doc) = sum over each ranking of 1 / (k + rank)
    Lower rank number (closer to 0) = higher contribution.
    """
    scores = {}

    for rank, idx in enumerate(dense_ranks):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    for rank, idx in enumerate(sparse_ranks):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    # Sort by combined score, descending
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused  # list of (chunk_idx, fused_score)

def hybrid_retrieve(query, top_k=3):
    dense_ranks = dense_retrieve(query, top_k=10)
    sparse_ranks = sparse_retrieve(query, top_k=10)
    fused = reciprocal_rank_fusion(dense_ranks, sparse_ranks)

    top_chunks = []
    print(f"\nQuery: {query}")
    for idx, score in fused[:top_k]:
        print(f"  Source: {chunk_sources[idx]} | Fused Score: {score:.4f}")
        print(f"  {all_chunks[idx][:150]}...\n")
        top_chunks.append(all_chunks[idx])
    return top_chunks

# --- Test it ---
hybrid_retrieve("When did Beyonce start becoming popular?")
hybrid_retrieve("What is solar energy?")
hybrid_retrieve("Who wrote To Kill a Mockingbird?")