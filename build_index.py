from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# --- Step 1: Rebuild the same corpus as before ---
dataset = load_dataset("rajpurkar/squad_v2")
train_data = dataset["train"]

NUM_ARTICLES = 20
unique_titles = list(dict.fromkeys(train_data["title"]))[:NUM_ARTICLES]

corpus = {}
for example in train_data:
    if example["title"] in unique_titles and example["title"] not in corpus:
        corpus[example["title"]] = example["context"]

# --- Step 2: Chunk each passage into smaller pieces ---
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

all_chunks = []       # the actual text chunks
chunk_sources = []    # which article title each chunk came from

for title, text in corpus.items():
    chunks = chunk_text(text)
    for c in chunks:
        all_chunks.append(c)
        chunk_sources.append(title)

print(f"Total chunks created: {len(all_chunks)}")

# --- Step 3: Embed all chunks ---
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding chunks...")
embeddings = model.encode(all_chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# --- Step 4: Build FAISS index ---
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} vectors of dimension {dimension}")

# --- Step 5: Save everything for later use ---
faiss.write_index(index, "faiss_index.bin")

with open("chunks_data.pkl", "wb") as f:
    pickle.dump({"chunks": all_chunks, "sources": chunk_sources}, f)

print("\nSaved: faiss_index.bin and chunks_data.pkl")