import json 
from collections import Counter 
from pathlib import Path

class BPETokenizer:
    def __init__(self):
        self.merges: list[tuple[int, int]] = []
        self.vocab_size = 256

    @staticmethod
    def _merge_pair(
            ids: list[int],
            pair: tuple[int, int], 
            new_id: int,
    ) -> list[int]:
        """Replace non-overlapping occurances of a pair, left to right."""
        result = []
        i = 0

        while i < len(ids):
            if (
                i + 1 < len(ids)
                and ids[i] == pair[0]
                and ids[i + 1] == pair[1]
            ):
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1

        return result

    def train(self, text: str, vocab_size: int) -> None:
        """Learn frequent adjacent-byte/token pairs from training text"""
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        # Convert text into the initial sequence of byte token IDs
        ids = list(text.encode("utf-8"))
        learned_merges = []

        # Each merge adds exactly one new token to the vocabulary 
        while 256 + len(learned_merges) < vocab_size:
            if len(ids) < 2:
                raise ValueError(
                    "Not enough adjacent pairs to reach the requested vocabulary size"
                )

            # Count every adjacent pair 
            pair_counts = Counter(zip(ids, ids[1:]))

            # Counter preserves insertion order
            best_pair = max(pair_counts, key=pair_counts.get)

            # replace the pair, then replace its non-overlapping occurences.
            new_id = 256 + len(learned_merges)
            learned_merges.append(best_pair)
            ids = self._merge_pair(ids, best_pair, new_id)

        self.merges = learned_merges
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs using the learned merge order"""
        ids = list(text.encode("utf-8"))

        for merge_index, pair in enumerate(self.merges):
            new_id = 256 + merge_index
            ids = self._merge_pair(ids, pair, new_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back into the UTF-8 text"""
        token_bytes = [bytes([i]) for i in range(256)]

        # Reconstruct the bytes represented by each learned token.
        for left_id, right_id in self.merges:
            if left_id >= len(token_bytes) or right_id >= len(token_bytes):
                raise ValueError("Invalid merge: referenced token does not exist.")

            merged_bytes = token_bytes[left_id] + token_bytes[right_id]
            token_bytes.append(merged_bytes)

        # Join bytes first: an individual token need not form valid UTF-8
        output = bytearray()

        for token_id in ids:
            if not isinstance(token_id, int) or not 0 <= token_id < len(token_bytes):
                raise ValueError(f"Invalid token ID: {token_id}")

            output.extend(token_bytes[token_id])
        return output.decode("utf-8")

    def save(self, path: str | Path) -> None:
        """Save vocabulary metadata and merges to a JSON file"""
        payload = {
            "vocab_size": self.vocab_size,
            "merges": [[left, right] for left, right in self.merges],
        }

        Path(path).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer and validate it's merge definitions"""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))

        tokenizer = cls()
        tokenizer.merges = [
            (int(pair[0]), int(pair[1]))
            for pair in payload["merges"]
        ]
        tokenizer.vocab_size = int(
            payload.get("vocab_size", 256 + len(tokenizer.merges))
        )

        if tokenizer.vocab_size != 256 + len(tokenizer.merges):
            raise ValueError("Vocabulary size does not match the merge count")

        # A merge can only refer to the base tokens or earlier merges.
        for merge_index, (left_id, right_id) in enumerate(tokenizer.merges):
            current_vocab_size = 256 + merge_index

            if not (
                0 <= left_id < current_vocab_size
                and 0 <= right_id < current_vocab_size
            ):
                raise ValueError(
                    f"Invalid merge at index {merge_index}:"
                    f"{(left_id, right_id)}"
                )
        return tokenizer    
