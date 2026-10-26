import math 
import csv
from pathlib import Path
from tokenizer import BPETokenizer

def main():
    data_path = Path("data/input.txt")
    tokenizer_path = Path("tokenizer.json")

    text = data_path.read_text(encoding="utf-8")
    split_at = int(len(text) * 0.9)

    train_text = text[:split_at]
    val_text = text[split_at:]

    tokenizer = BPETokenizer.load(tokenizer_path)
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    vocab_size = tokenizer.vocab_size

    # Count adjacent-token transitions in the training split.
    counts = [[0] * vocab_size for _ in range(vocab_size)]

    for a, b in zip(train_ids[:-1], train_ids[1:]):
        counts[a][b] += 1

    # Add-one smoothing: give every possible transition a count of 1.
    row_totals = [sum(row) + vocab_size for row in counts]

    total_loss = 0.0
    total_tokens = 0

    # Score the next token using only the previous token.
    for a, b in zip(val_ids[:-1], val_ids[1:]):
        probability = (counts[a][b] + 1) / row_totals[a]
        total_loss -= math.log(probability)
        total_tokens += 1

    val_loss = total_loss / total_tokens

    print(f"Bigram validation loss: {val_loss:.4f} nats/token")

    # Record the baseline in runs.csv
    runs_path = Path("runs.csv")
    fields = [
        "run_id", "n_layer", "n_head", "n_embd", "block_size",
        "max_lr", "max_steps", "ablation", "params",
        "best_val_loss", "final_train_loss", "tokens_per_sec",
        "wall_time_min",
    ]

    row = {
        "run_id": "bigram",
        "n_layer": 0,
        "n_head": 0,
        "n_embd": 0,
        "block_size": 1,
        "max_lr": 0,
        "max_steps": 0,
        "ablation": "none",
        "params": vocab_size * vocab_size,
        "best_val_loss": val_loss,
        "final_train_loss": "",
        "tokens_per_sec": "",
        "wall_time_min": "",
    }

    write_header = not runs_path.exists() or runs_path.stat().st_size == 0
    with runs_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()

