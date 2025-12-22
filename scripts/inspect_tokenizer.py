import json

with open("models/llama3-8b/tokenizer.json", "r") as f:
    data = json.load(f)

print("Model Type:", data.get("model", {}).get("type"))
vocab = data.get("model", {}).get("vocab", {})
merges = data.get("model", {}).get("merges", [])

print("\nFirst 10 vocab items:")
for k, v in list(vocab.items())[:10]:
    print(f"'{k}': {v}")

print("\nFirst 10 merges:")
for m in merges[:10]:
    print(f"'{m}'")

if "pre_tokenizer" in data:
    print("\nPre-tokenizer:", json.dumps(data["pre_tokenizer"], indent=2))
