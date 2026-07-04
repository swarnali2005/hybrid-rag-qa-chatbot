from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from dotenv import load_dotenv
from groq import Groq

# --- Setup ---
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

faiss_index = faiss.read_index("faiss_index.bin")
with open("chunks_data.pkl", "rb") as f:
    faiss_data = pickle.load(f)
    all_chunks = faiss_data["chunks"]
    chunk_sources = faiss_data["sources"]

with open("bm25_index.pkl", "rb") as f:
    bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def dense_retrieve(query, top_k=10):
    query_vector = embed_model.encode([query]).astype("float32")
    distances, indices = faiss_index.search(query_vector, top_k)
    return list(indices[0])

def sparse_retrieve(query, top_k=10):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return ranked_indices

def reciprocal_rank_fusion(dense_ranks, sparse_ranks, k=60, dense_weight=2.0, sparse_weight=1.0):
    scores = {}
    for rank, idx in enumerate(dense_ranks):
        scores[idx] = scores.get(idx, 0) + dense_weight * (1 / (k + rank + 1))
    for rank, idx in enumerate(sparse_ranks):
        scores[idx] = scores.get(idx, 0) + sparse_weight * (1 / (k + rank + 1))
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused

def hybrid_retrieve(query, top_k=5):
    dense_ranks = dense_retrieve(query, top_k=10)
    sparse_ranks = sparse_retrieve(query, top_k=10)
    fused = reciprocal_rank_fusion(dense_ranks, sparse_ranks)
    top_chunks = [all_chunks[idx] for idx, score in fused[:top_k]]
    return top_chunks

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Answer the question using ONLY the context below.
Give a SHORT answer — just the key fact/phrase, not a full sentence, no explanation.
If the answer is not in the context, say exactly: "I don't have enough information to answer this."

Context:
{context}

Question: {query}

Short Answer:"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=30
    )
    return response.choices[0].message.content.strip()

def hybrid_rag_pipeline(query):
    chunks = hybrid_retrieve(query)
    answer = generate_answer(query, chunks)
    print(f"\nQuestion: {query}")
    print(f"Answer: {answer}")

# --- Test it ---
hybrid_rag_pipeline("When did Beyonce start becoming popular?")
hybrid_rag_pipeline("What is solar energy?")
hybrid_rag_pipeline("Who wrote To Kill a Mockingbird?")
hybrid_rag_pipeline("What is the capital of France?")