import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import pickle
import os
from dotenv import load_dotenv
from groq import Groq

# --- Page setup ---
st.set_page_config(page_title="Hybrid RAG Q&A Chatbot", page_icon="🤖")
st.title("🤖 Hybrid RAG Q&A Chatbot")
st.caption("Ask questions about: Beyoncé, Chopin, iPod, Zelda, Spectre, Sichuan Earthquake, New York City, To Kill a Mockingbird, Solar Energy, Kanye West, Buddhism, American Idol, Dog, Olympics, Genome, and more.")

# --- Load everything once (cached so it doesn't reload on every interaction) ---
@st.cache_resource
def load_resources():
    load_dotenv()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    faiss_index = faiss.read_index("faiss_index.bin")
    with open("chunks_data.pkl", "rb") as f:
        faiss_data = pickle.load(f)
        all_chunks = faiss_data["chunks"]

    with open("bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
        bm25 = bm25_data["bm25"]

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return client, faiss_index, all_chunks, bm25, embed_model, reranker

client, faiss_index, all_chunks, bm25, embed_model, reranker = load_resources()

# --- Pipeline functions ---
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

def hybrid_retrieve_with_rerank(query, fusion_top_k=5, final_top_k=3):
    dense_ranks = dense_retrieve(query, top_k=10)
    sparse_ranks = sparse_retrieve(query, top_k=10)
    fused = reciprocal_rank_fusion(dense_ranks, sparse_ranks)
    candidate_indices = [idx for idx, score in fused[:fusion_top_k]]
    candidate_chunks = [all_chunks[idx] for idx in candidate_indices]

    pairs = [[query, chunk] for chunk in candidate_chunks]
    rerank_scores = reranker.predict(pairs)

    scored_chunks = list(zip(candidate_chunks, rerank_scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, score in scored_chunks[:final_top_k]]
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

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_question = st.chat_input("Ask a question...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved_chunks = hybrid_retrieve_with_rerank(user_question)
            answer = generate_answer(user_question, retrieved_chunks)
            st.write(answer)

            with st.expander("📄 View retrieved context"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.markdown(f"**Chunk {i+1}:** {chunk[:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})