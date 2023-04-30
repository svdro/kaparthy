import torch
from torch import nn, Tensor
from torch.nn import functional as F

from typing import Optional


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        # self.tril =torch.tril(torch.ones(block_size, block_size))

    def forward(self, x):
        _, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute attention scores
        assert isinstance(self.tril, torch.Tensor)  # to shut up auto-complete
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        return wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)


class MultiHeadAttention(nn.Module):
    """Multi-head attention"""

    def __init__(self, n_heads: int, head_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_heads * head_size, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """Feed-forward layer: linear layer followed by ReLU"""

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # projection
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_heads: int, n_embd: int, block_size: int):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, vocab_size: int, n_heads: int, n_embd: int, block_size: int = 128
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_heads, n_embd, block_size),
            Block(n_heads, n_embd, block_size),
            Block(n_heads, n_embd, block_size),
            nn.LayerNorm(n_embd),
        )
        # self.sa_heads = MultiHeadAttention(
        # n_heads, n_embd // n_heads, n_embd, block_size
        # )
        # self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None) -> tuple[Tensor, Optional[Tensor]]:
        """
        idx: (bsz, block_size)
        targets: (bsz, block_size)
        return logits: (bsz, block_size, vocab_size)
        """
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb  # (B, T, C)
        # x = self.sa_heads(x)  # (B, T, C)
        # x = self.ffwd(x)  # (B, T, C)

        x = self.blocks(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: Tensor, max_new_tokens: int):
        block_size, _ = self.position_embedding_table.weight.shape

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, _ = self(idx_cond, None)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None) -> tuple[Tensor, Optional[Tensor]]:
        """
        idx: (bsz, block_size)
        targets: (bsz, block_size)
        return logits: (bsz, block_size, vocab_size)
        """
        logits = self.token_embedding_table(idx)
        B, T, C = logits.shape

        loss = None
        if targets is not None:
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, _ = self(idx, None)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx
