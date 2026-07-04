from datasets import load_dataset

# Load SQuAD 2.0
dataset = load_dataset("rajpurkar/squad_v2")
train_data = dataset["train"]

# Pick a fixed number of unique articles (titles)
NUM_ARTICLES = 20
unique_titles = list(dict.fromkeys(train_data["title"]))[:NUM_ARTICLES]

print(f"Selected {len(unique_titles)} articles:")
for t in unique_titles:
    print(" -", t)

# Build our knowledge base: one unique context per article
# (SQuAD repeats the same context for multiple questions, so we deduplicate)
corpus = {}
for example in train_data:
    if example["title"] in unique_titles and example["title"] not in corpus:
        corpus[example["title"]] = example["context"]

print(f"\nBuilt corpus with {len(corpus)} unique passages.")
print("\n--- Example passage ---")
sample_title = unique_titles[0]
print(f"Title: {sample_title}")
print(corpus[sample_title][:300], "...")