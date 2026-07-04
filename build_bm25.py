from datasets import load_dataset
from rank_bm25 import BM25Okapi
import pickle

# --- Rebuild the same corpus/chunks as before ---
dataset = load_dataset("rajpurkar/squad_v2")
train_data = dataset["train"]

NUM_ARTICLES = 20
unique_titles = list(dict.fromkeys(train_data["title"]))[:NUM_ARTICLES]

corpus = {}
for example in train_data:
    if example["title"] in unique_titles and example["title"] not in corpus:
        corpus[example["title"]] = example["context"]

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

all_chunks = []
chunk_sources = []

for title, text in corpus.items():
    chunks = chunk_text(text)
    for c in chunks:
        all_chunks.append(c)
        chunk_sources.append(title)

# --- Build BM25 index ---
# BM25 needs tokenized (word-split) text
tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)

print(f"BM25 index built with {len(all_chunks)} chunks.")

# --- Save BM25 index + chunk data (reuse same chunks/sources as FAISS) ---
with open("bm25_index.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "chunks": all_chunks, "sources": chunk_sources}, f)

print("Saved: bm25_index.pkl")

# --- Quick test ---
def bm25_retrieve(query, top_k=3):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    print(f"\nQuery: {query}")
    for idx in top_indices:
        print(f"  Source: {chunk_sources[idx]} | Score: {scores[idx]:.4f}")
        print(f"  {all_chunks[idx][:150]}...\n")

bm25_retrieve("When did Beyonce start becoming popular?")
bm25_retrieve("What is solar energy?")