from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import re
import string
import time
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
    return [all_chunks[idx] for idx, score in fused[:top_k]]

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

def hybrid_rag_answer(query):
    chunks = hybrid_retrieve(query)
    return generate_answer(query, chunks)

# --- Text normalization ---
def normalize_text(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s

def is_abstained(answer):
    return "don't have enough information" in answer.lower() or "not enough information" in answer.lower()

def compute_em(prediction, gold_answers):
    pred_norm = normalize_text(prediction)
    return int(any(normalize_text(g) in pred_norm or pred_norm in normalize_text(g) for g in gold_answers))

# --- Load same 15 answerable + 10 unanswerable SQuAD questions ---
dataset = load_dataset("rajpurkar/squad_v2")
train_data = dataset["train"]

NUM_ARTICLES = 20
unique_titles = list(dict.fromkeys(train_data["title"]))[:NUM_ARTICLES]

answerable_qs = []
unanswerable_qs = []

for example in train_data:
    if example["title"] not in unique_titles:
        continue
    if len(example["answers"]["text"]) > 0:
        if len(answerable_qs) < 15:
            answerable_qs.append(example)
    else:
        if len(unanswerable_qs) < 10:
            unanswerable_qs.append(example)
    if len(answerable_qs) >= 15 and len(unanswerable_qs) >= 10:
        break

# --- Hand-written paraphrase "stress-test" questions ---
paraphrase_qs = [
    {"question": "When did Beyonce first gain recognition?", "answers": {"text": ["late 1990s"]}},
    {"question": "What group made Beyonce famous initially?", "answers": {"text": ["Destiny's Child"]}},
    {"question": "How does the sun's energy get used by technology?", "answers": {"text": ["photovoltaics", "solar heating", "solar thermal energy"]}},
    {"question": "Which novelist is behind To Kill a Mockingbird?", "answers": {"text": ["Harper Lee"]}},
    {"question": "What breed of animal is commonly kept as a pet and descended from wolves?", "answers": {"text": ["dog", "Dog"]}},
]

print(f"Evaluating on {len(answerable_qs)} SQuAD answerable + {len(paraphrase_qs)} paraphrase + {len(unanswerable_qs)} unanswerable questions\n")

DELAY = 2.5

# --- Run on SQuAD answerable ---
correct_squad = 0
for i, ex in enumerate(answerable_qs):
    pred = hybrid_rag_answer(ex["question"])
    gold = ex["answers"]["text"]
    em = compute_em(pred, gold)
    correct_squad += em
    status = "RIGHT" if em else "WRONG"
    print(f"[SQuAD {status}] ({i+1}/{len(answerable_qs)}) Q: {ex['question']}")
    print(f"        Gold: {gold} | Predicted: {pred}\n")
    time.sleep(DELAY)

# --- Run on paraphrase stress-test set ---
correct_paraphrase = 0
for i, ex in enumerate(paraphrase_qs):
    pred = hybrid_rag_answer(ex["question"])
    gold = ex["answers"]["text"]
    em = compute_em(pred, gold)
    correct_paraphrase += em
    status = "RIGHT" if em else "WRONG"
    print(f"[PARAPHRASE {status}] ({i+1}/{len(paraphrase_qs)}) Q: {ex['question']}")
    print(f"        Gold: {gold} | Predicted: {pred}\n")
    time.sleep(DELAY)

# --- Run on unanswerable ---
hallucinated = 0
for i, ex in enumerate(unanswerable_qs):
    pred = hybrid_rag_answer(ex["question"])
    if not is_abstained(pred):
        hallucinated += 1
        print(f"[HALLUCINATED] ({i+1}/{len(unanswerable_qs)}) Q: {ex['question']}")
        print(f"        Predicted: {pred}\n")
    else:
        print(f"[ABSTAINED OK] ({i+1}/{len(unanswerable_qs)}) Q: {ex['question']}\n")
    time.sleep(DELAY)

# --- Report ---
total_answerable = len(answerable_qs) + len(paraphrase_qs)
total_correct = correct_squad + correct_paraphrase

print("=" * 60)
print(f"HYBRID RAG RESULTS (dense_weight=2.0, top_k=5)")
print(f"SQuAD accuracy: {correct_squad}/{len(answerable_qs)} = {correct_squad/len(answerable_qs)*100:.1f}%")
print(f"Paraphrase (stress-test) accuracy: {correct_paraphrase}/{len(paraphrase_qs)} = {correct_paraphrase/len(paraphrase_qs)*100:.1f}%")
print(f"Overall answerable accuracy: {total_correct}/{total_answerable} = {total_correct/total_answerable*100:.1f}%")
print(f"Hallucination rate: {hallucinated}/{len(unanswerable_qs)} = {hallucinated/len(unanswerable_qs)*100:.1f}%")
print("=" * 60)