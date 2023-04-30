import torch
from torch import nn, Tensor
from torch.nn import functional as F

from typing import Optional


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

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
