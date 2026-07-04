from datasets import load_dataset

dataset = load_dataset("rajpurkar/squad_v2")

print(dataset)
print("\n--- Sample entry ---")
print(dataset["train"][0])