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

index = faiss.read_index("faiss_index.bin")
with open("chunks_data.pkl", "rb") as f:
    data = pickle.load(f)
    all_chunks = data["chunks"]
    chunk_sources = data["sources"]

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, top_k=3):
    query_vector = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    return [all_chunks[idx] for idx in indices[0]]

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

def naive_rag_answer(query):
    chunks = retrieve(query)
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

# --- Load SQuAD and build a SMALLER eval set ---
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

print(f"Evaluating on {len(answerable_qs)} answerable + {len(unanswerable_qs)} unanswerable questions\n")

# --- Run evaluation WITH a delay between calls to avoid rate limits ---
DELAY = 2.5  # seconds between API calls

correct = 0
total_answerable = len(answerable_qs)
for i, ex in enumerate(answerable_qs):
    pred = naive_rag_answer(ex["question"])
    gold = ex["answers"]["text"]
    em = compute_em(pred, gold)
    correct += em
    status = "RIGHT" if em else "WRONG"
    print(f"[{status}] ({i+1}/{total_answerable}) Q: {ex['question']}")
    print(f"        Gold: {gold} | Predicted: {pred}\n")
    time.sleep(DELAY)

hallucinated = 0
total_unanswerable = len(unanswerable_qs)
for i, ex in enumerate(unanswerable_qs):
    pred = naive_rag_answer(ex["question"])
    if not is_abstained(pred):
        hallucinated += 1
        print(f"[HALLUCINATED] ({i+1}/{total_unanswerable}) Q: {ex['question']}")
        print(f"        Predicted: {pred}\n")
    else:
        print(f"[ABSTAINED OK] ({i+1}/{total_unanswerable}) Q: {ex['question']}\n")
    time.sleep(DELAY)

# --- Report ---
print("=" * 50)
print(f"Accuracy on answerable questions: {correct}/{total_answerable} = {correct/total_answerable*100:.1f}%")
print(f"Hallucination rate on unanswerable questions: {hallucinated}/{total_unanswerable} = {hallucinated/total_unanswerable*100:.1f}%")
print("=" * 50)