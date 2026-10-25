from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class GPTConfig:
    vocab_size: int = 1024
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

class CausalSelfAttentionK(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Project each input vector into Query, Key, and Value
        self.qkv = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=False,
        )

        # Project the combined attention heads back to embedding size
        self.out_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape

        # [B, T, 3*d] -> [B, T, 3, H, head_dim]
        qkv = self.qkv(x).view(
            batch_size,
            seq_len,
            3,
            self.n_head,
            self.head_dim,
        )

        # Separate Q, K, V and arrange each as [B, H, T, head_dim]
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        # Similarity scores between every query and key 
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Prevent each position from attending to future positions
        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            )
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # Turn allowed scores into attention probabilities 
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # Combine values using the attetion probabilites 
        output = weights @ v

        # merge heads
        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            n_embd,
        )

        output = self.out_proj(output)
        return self.resid_dropout(output)

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.fc1 = nn.Linear(
            config.n_embd,
            4 * config.n_embd,
            bias=False,
        )
        self.fc2 = nn.Linear(
            4 * config.n_embd, 
            config.n_embd,
            bias=False,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return self.dropout(x)

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionK(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # Convert tokens into learning vectors.
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        # Give each position a learned vector.
        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd,
        )

        # Apply dropout to the combined embeddings.
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layer)
        ])

        self.final_norm = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )

        # The output weights to the token embedding weights
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = idx.shape

        if seq_len > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")

        positions = torch.arange(seq_len, device=idx.device)
        token_embeddings = self.token_embedding(idx)
        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    def num_params(self) -> int:
        return sum(
            parameter.numel()
            for parameter in {id(p): p for p in self.parameters()}.values()
        )
