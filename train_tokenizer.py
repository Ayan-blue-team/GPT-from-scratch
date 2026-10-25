from pathlib import Path
from tokenizer import BPETokenizer

dataset_path = Path("data/input.txt")

if not dataset_path.exists():
    raise FileNotFoundError("Dataset missing.")

text = dataset_path.read_text(encoding="utf-8")

# Reproduce the assignment's character-based 90/10 split.
split_index = int(len(text) * 0.9)
train_text = text[:split_index]
holdout_text = text[split_index:]

print(f"Training characters: {len(train_text):,}")
print(f"Holdout characters: {len(holdout_text):,}")
print("Training BPE tokenizer with vocabulary size 1024")

tokenizer = BPETokenizer()
tokenizer.train(train_text, vocab_size=1024)

# Save the final tokenizer.
tokenizer.save("tokenizer.json")

# Verify the required vocabulary and merge count.
assert tokenizer.vocab_size == 1024
assert len(tokenizer.merges) == 768

# Check that training and holdout text can both be reconstructed.
train_sample = train_text[:2000]
holdout_sample = holdout_text[:2000]

assert tokenizer.decode(tokenizer.encode(train_sample)) == train_sample
assert tokenizer.decode(tokenizer.encode(holdout_sample)) == holdout_sample

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Number of merges: {len(tokenizer.merges)}")
print("Training sample round trip: passed")
print("Holdout sample round trip: passed")
print("Saved tokenizer.json")
