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
    ablation: str = "none"


class CausalSelfAttentionK(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=False
        )
        self.out_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=False
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor]
    ]:
        batch_size, seq_len, n_embd = x.shape

        qkv = self.qkv(x).view(
            batch_size, seq_len, 3, self.n_head, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        past_len = 0
        if cache is not None:
            past_k, past_v = cache
            past_len = past_k.size(-2)
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present = (k, v) if use_cache else None

        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Cached queries can attend to all past keys and to earlier
        # positions in the current chunk, but never to future positions.
        key_positions = torch.arange(k.size(-2), device=x.device)
        query_positions = past_len + torch.arange(seq_len, device=x.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, n_embd
        )
        output = self.out_proj(output)
        output = self.resid_dropout(output)

        if use_cache:
            return output, present
        return output


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

        self.post_ln = getattr(config, "ablation", "none") == "post_ln"

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        if self.post_ln:
            # A2: Post-LN
            if use_cache:
                attn_out, present = self.attn(
                    x, cache=cache, use_cache=True
                )
            else:
                attn_out = self.attn(x)
                present = None

            x = self.ln1(x + attn_out)
            x = self.ln2(x + self.mlp(x))

        else:
            # Default: Pre-LN
            if use_cache:
                attn_out, present = self.attn(
                    self.ln1(x), cache=cache, use_cache=True
                )
                x = x + attn_out
            else:
                x = x + self.attn(self.ln1(x))
                present = None

            x = x + self.mlp(self.ln2(x))

        if use_cache:
            return x, present
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

        self.use_position_embedding = (
            getattr(config, "ablation", "none")
            != "no_position_embedding"
        )

        self.position_embedding = (
            nn.Embedding(config.block_size, config.n_embd)
            if self.use_position_embedding
            else None
        )

        # Apply dropout to the embeddings.
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layer)
        ])

        self.final_norm = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        # Tie output weights to the token embedding weights.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: list[
            tuple[torch.Tensor, torch.Tensor] | None
        ] | None = None,
        use_cache: bool = False,
    ):
        batch_size, seq_len = idx.shape

        past_len = 0
        if cache is not None:
            if len(cache) != len(self.blocks):
                raise ValueError(
                    "Cache must contain one entry per block"
                )

            if cache[0] is not None:
                past_len = cache[0][0].size(-2)

        if past_len + seq_len > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")

        positions = torch.arange(
            past_len,
            past_len + seq_len,
            device=idx.device,
        )

        # Token embeddings are always present.
        x = self.token_embedding(idx)

        # A1 disables the addition of positional embeddings.
        if self.use_position_embedding:
            x = x + self.position_embedding(positions)

        x = self.embedding_dropout(x)

        new_cache = []

        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None

            if use_cache:
                x, present = block(
                    x,
                    cache=layer_cache,
                    use_cache=True,
                )
                new_cache.append(present)
            else:
                x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        if use_cache:
            return logits, loss, new_cache

        return logits, loss

    def num_params(self) -> int:
        return sum(
            parameter.numel()
            for parameter in {id(p): p for p in self.parameters()}.values()
        )
