
from pathlib import Path
from urllib.request import urlopen

URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

dataset_path = data_dir / "input.txt"

if not dataset_path.exists():
    with urlopen(URL, timeout=30) as response:
        text = response.read().decode("utf-8")

    dataset_path.write_text(text, encoding="utf-8")
else:
    text = dataset_path.read_text(encoding="utf-8")

# Split by character position: first 90% for training, final 10% holdout.
split_index = int(len(text) * 0.9)

train_text = text[:split_index]
holdout_text = text[split_index:]

print(f"Total: {len(text):,}")
print(f"Training: {len(train_text):,}")
print(f"Holdout: {len(holdout_text):,}")
print(f"Dataset saved to: {dataset_path}")
