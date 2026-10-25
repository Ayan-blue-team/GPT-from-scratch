import json 
import random 
from pathlib import Path
import csv
import time
import torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer
import math

def load_data_and_tokenizer():
    data_path = Path("data/input.txt")
    okenizer_path = Path("data/input.txt")
    tokenizer_path = Path("tokenizer.json")
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            "Dataset not found."
        )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            "Tokenizer not found."
        )

    text = data_path.read_text(encoding="utf=8")
    split_at = int(len(text) * 0.9)

    train_text = text[:split_at]
    val_text = text[split_at:]

    tokenizer = BPETokenizer.load(tokenizer_path)

    train_cache = cache_dir / "train_ids.pt"
    val_cache = cache_dir / "val_ids.pt"

    if train_cache.exists() and val_cache.exists():
        train_ids = torch.load(
            train_cache, map_location="cpu", weights_only=True
        )    
        val_ids = torch.load(
            val_cache, map_location="cpu", weights_only=True
        )
    else:
        train_ids = torch.tensor(
            tokenizer.encode(train_text), dtype=torch.long
        )
        val_ids = torch.tensor(
            tokenizer.encode(val_text), dtype=torch.long
        )
        torch.save(train_ids, train_cache)
        torch.save(val_ids, val_cache)

    return tokenizer, train_ids, val_ids


def get_batch(data, batch_size, block_size, device, rng):
    max_start = len(data) - block_size - 1
    if max_start < 0:
        raise ValueError("Not enough tokens to create a training batch")

    starts = [
        rng.randint(0, max_start)
        for _ in range(batch_size)
    ]

    x = torch.stack([
        data[start : start + block_size]
        for start in starts
    ])
    y = torch.stack([
        data[start + 1 : start + block_size + 1]
        for start in starts
    ])

    return x.to(device), y.to(device)


def create_optimizer(model, max_lr):
    decay_params = []
    no_decay_params = []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        if parameter.dim() >= 2:
            decay_params.append(parameter)
        else:
            no_decay_params.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weights_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=max_lr,
        betas=(0.9, 0.95),
    )
    return optimizer

def get_lr(step, max_lr, max_steps):
    warmup_steps = 137

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    progress = min(
        (step - warmup_steps) / (max_steps - warmup_steps),
        1.0,
    )

    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = 0.1 * max_lr

    return min_lr + (max_lr - min_lr) * cosine_factor

def evaluate(model, data, batch_size, block_size, device, rng):
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(50):
            x, y = get_batch(data, batch_size, block_size, device, rng)
            _, loss = model(x, y)
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)

def train(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    random_seed = config["seed"]
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, train_ids, val_ids = load_data_and_tokenizer()

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
    )
    model = GPT(model_config).to(device)
    optimizer = create_optimizer(model, config["max_lr"])

    run_id = config["run_id"]
    max_steps = config["max_steps"]
    batch_size = config["batch_size"]
    block_size = config["block_size"]
    eval_interval = config["eval_interval"]
    max_lr = config["max_lr"]

    log_dir = Path("logs")
    checkpoint_dir = Path("checkpoints")
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    log_path = log_dir / f"{run_id}.csv"
    checkpoint_path = checkpoint_dir / f"{run_id}_best.pt"

    log_fields = ["step", "train_loss", "val_loss", "lr", "tokens_seen"]
    run_fields = [
        "run_id", "n_layer", "n_head", "n_embd", "block_size",
        "max_lr", "max_steps", "ablation", "params",
        "best_val_loss", "final_train_loss", "tokens_per_sec",
        "wall_time_min",
    ]

    batch_rng = random.Random(random_seed + 1)
    eval_rng = random.Random(random_seed + 2)
    start_time = time.time()
    tokens_seen = 0
    best_val_loss = float("inf")
    final_train_loss = float("nan")

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=log_fields)
        writer.writeheader()

        for step in range(max_steps):
            lr = get_lr(step, max_lr, max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            model.train()
            x, y = get_batch(
                train_ids, batch_size, block_size, device, batch_rng
            )
            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            final_train_loss = loss.item()
            tokens_seen += batch_size * block_size

            if (step + 1) % eval_interval == 0 or step == 0:
                val_loss = evaluate(
                    model, val_ids, batch_size, block_size, device, eval_rng
                )

                writer.writerow({
                    "step": step + 1,
                    "train_loss": final_train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                    "tokens_seen": tokens_seen,
                })
                log_file.flush()

                print(
                    f"Step {step + 1}/{max_steps} | "
                    f"train loss {final_train_loss:.4f} | "
                    f"val loss {val_loss:.4f} | lr {lr:.6g}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "model": model.state_dict(),
                        "config": model_config.__dict__,
                        "step": step + 1,
                        "best_val_loss": best_val_loss,
                    }, checkpoint_path)

    wall_time = time.time() - start_time
    tokens_per_sec = tokens_seen / wall_time if wall_time else 0.0

    run_row = {
        "run_id": run_id,
        "n_layer": config["n_layer"],
        "n_head": config["n_head"],
        "n_embd": config["n_embd"],
        "block_size": block_size,
        "max_lr": max_lr,
        "max_steps": max_steps,
        "ablation": config["ablation"],
        "params": model.num_params(),
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "tokens_per_sec": tokens_per_sec,
        "wall_time_min": wall_time / 60,
    }

    runs_path = Path("runs.csv")
    write_header = not runs_path.exists() or runs_path.stat().st_size == 0
    with runs_path.open("a", newline="", encoding="utf-8") as runs_file:
        writer = csv.DictWriter(runs_file, fieldnames=run_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(run_row)

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
