import torch
import json
import time
from pathlib import Path

from model import GPT, GPTConfig
from tokenizer import BPETokenizer

def generate(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
    seed: int | None = None,
) -> list[int]:
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")

    block_size = model.config.block_size
    if len(prompt_ids) + max_new_tokens > block_size:
        raise ValueError("Prompt plus generated tokens exceeds block_size.")

    device = next(model.parameters()).device
    result = list(prompt_ids)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            cache = None

            # Process the full prompt once to populate the cache.
            if use_cache:
                context = torch.tensor(
                    [result], dtype=torch.long, device=device
                )
                logits, _, cache = model(context, use_cache=True)
                next_logits = logits[0, -1, :]

            for _ in range(max_new_tokens):
                if not use_cache:
                    context = torch.tensor(
                        [result], dtype=torch.long, device=device
                    )
                    logits, _ = model(context)
                    next_logits = logits[0, -1, :]

                if temperature == 0:
                    next_token = int(torch.argmax(next_logits).item())
                else:
                    scores = next_logits / temperature

                    # Apply top-k filtering first.
                    if top_k is not None:
                        k = min(top_k, scores.numel())
                        threshold = torch.topk(scores, k).values[-1]
                        scores = scores.masked_fill(
                            scores < threshold, float("-inf")
                        )

                    probs = torch.softmax(scores, dim=-1)

                    # Keep the smallest sorted prefix with cumulative
                    # probability >= top_p.
                    if top_p is not None:
                        sorted_probs, sorted_indices = torch.sort(
                            probs, descending=True
                        )
                        cumulative = torch.cumsum(sorted_probs, dim=-1)
                        remove = cumulative - sorted_probs >= top_p
                        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                        filtered = torch.zeros_like(probs)
                        probs = filtered.scatter(
                            0, sorted_indices, sorted_probs
                        )
                        probs = probs / probs.sum()

                    next_token = int(
                        torch.multinomial(
                            probs, num_samples=1, generator=generator
                        ).item()
                    )

                result.append(next_token)

                # Feed only the newly generated token into the cache.
                if use_cache and len(result) < len(prompt_ids) + max_new_tokens:
                    newest = torch.tensor(
                        [[next_token]], dtype=torch.long, device=device
                    )
                    logits, _, cache = model(
                        newest, cache=cache, use_cache=True
                    )
                    next_logits = logits[0, -1, :]

    finally:
        model.train(was_training)

    return result

if __name__ == "__main__":
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the existing trained checkpoint.
    checkpoint = torch.load(
        "checkpoints/M_best.pt",
        map_location=device,
        weights_only=False,
    )

    # Support common checkpoint formats.
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load the model configuration.
    with open("configs/M.json", encoding="utf-8") as f:
        saved_config = json.load(f)

    config_keys = {
        "vocab_size", "block_size", "n_layer",
        "n_head", "n_embd", "dropout",
    }

    checkpoint_config = checkpoint.get("config", {})
    if hasattr(checkpoint_config, "__dict__"):
        checkpoint_config = vars(checkpoint_config)
    if not isinstance(checkpoint_config, dict):
        checkpoint_config = {}

    config_values = {
        key: checkpoint_config[key]
        for key in config_keys
        if key in checkpoint_config
    }

    for key in config_keys:
        if key in saved_config:
            config_values.setdefault(key, saved_config[key])

    config_values.setdefault("vocab_size", 1024)
    config = GPTConfig(**config_values)

    model = GPT(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load the trained tokenizer.
    tokenizer = BPETokenizer.load("tokenizer.json")

    prompt_ids = tokenizer.encode("ROMEO:")
    if not prompt_ids:
        raise ValueError("The prompt produced no token IDs.")

    # Keep the benchmark within the model's context window.
    new_tokens = min(120, config.block_size - len(prompt_ids))
    if new_tokens < 1:
        raise ValueError("Prompt is too long for the model context.")

    # Confirm cached and uncached greedy generation agree.
    cached_ids = generate(
        model, prompt_ids, new_tokens,
        temperature=0, use_cache=True,
    )
    uncached_ids = generate(
        model, prompt_ids, new_tokens,
        temperature=0, use_cache=False,
    )

    if cached_ids != uncached_ids:
        raise AssertionError(
            "Cached and uncached greedy generation differ."
        )

    print("Cached and uncached greedy outputs match.")

    # Benchmark 120 new tokens where the context allows it.
    # Warm up both paths first.
    bench_tokens = min(120, config.block_size - len(prompt_ids))
    for cache_enabled in (True, False):
        generate(
            model, prompt_ids, min(4, bench_tokens),
            temperature=0, use_cache=cache_enabled,
        )

    timings = {}
    for cache_enabled in (True, False):
        start = time.perf_counter()
        generate(
            model, prompt_ids, bench_tokens,
            temperature=0, use_cache=cache_enabled,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        timings[cache_enabled] = bench_tokens / elapsed

    cached_speed = timings[True]
    uncached_speed = timings[False]
    print(f"Device: {device}")
    print(f"New tokens benchmarked: {bench_tokens}")
    print(f"Cached speed:   {cached_speed:.2f} tokens/sec")
    print(f"Uncached speed: {uncached_speed:.2f} tokens/sec")
    print(f"Speed-up:       {cached_speed / uncached_speed:.2f}x")

    # Generate and save a readable sample.
    sample_ids = generate(
        model,
        prompt_ids,
        min(200, config.block_size - len(prompt_ids)),
        temperature=0.8,
        top_k=40,
        seed=42,
        use_cache=True,
    )
    sample_text = tokenizer.decode(sample_ids)

    Path("samples.txt").write_text(
        sample_text + "\n",
        encoding="utf-8",
    )
    print("Generated sample saved to samples.txt")

