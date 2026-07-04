from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from dotenv import load_dotenv
from groq import Groq

# --- Load environment variables (API key) ---
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Load saved index and chunk data ---
index = faiss.read_index("faiss_index.bin")

with open("chunks_data.pkl", "rb") as f:
    data = pickle.load(f)
    all_chunks = data["chunks"]
    chunk_sources = data["sources"]

# --- Load embedding model ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, top_k=3):
    query_vector = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    retrieved_chunks = [all_chunks[idx] for idx in indices[0]]
    return retrieved_chunks

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question based only on the context below.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def naive_rag_pipeline(query):
    retrieved_chunks = retrieve(query)
    answer = generate_answer(query, retrieved_chunks)
    print(f"\nQuestion: {query}")
    print(f"Answer: {answer}")

# --- Test it ---
naive_rag_pipeline("When did Beyonce start becoming popular?")
naive_rag_pipeline("What is solar energy?")
naive_rag_pipeline("Who wrote To Kill a Mockingbird?")
naive_rag_pipeline("What is the capital of France?")  # should say "I don't have enough info" since it's not in our corpus