import torch
from torch import nn, Tensor
from models import BigramLanguageModel
from dataclasses import dataclass


def load_data(path: str = "input.txt"):
    with open(path, "r") as f:
        return f.read()


class Tokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        print(f"({self.vocab_size}) unique chars in data of len ({len(text)})")
        print(f"chars: {''.join(self.chars)}".encode("unicode_escape"))
        print(f"encode('hello'): {self.encode('hello')}")
        print(f"decode('hello'): {self.decode(self.encode('hello'))}")

    def encode(self, s: str):
        return [self.stoi[ch] for ch in s]

    def decode(self, l: list):
        return "".join([self.itos[i] for i in l])


class Dataset:
    def __init__(self, data: Tensor, bsz: int, block_size: int):
        self.data = data
        self.block_size = block_size
        self.bsz = bsz
        self.n = len(self.data)

    def get_batch(self) -> tuple[Tensor, Tensor]:
        inds = torch.randint(self.n - self.block_size, (self.bsz,))
        x = torch.stack([self.data[i : i + self.block_size] for i in inds])
        y = torch.stack([self.data[i + 1 : i + self.block_size + 1] for i in inds])
        return x, y


@torch.no_grad()
def estimate_loss(model: nn.Module, ds: Dataset, eval_steps: int) -> float:
    model.eval()

    losses = torch.zeros(eval_steps)
    for k in range(eval_steps):
        xb, yb = ds.get_batch()
        _, loss = model(xb, yb)
        losses[k] = loss.item()

    model.train()
    return losses.mean().item()


@dataclass
class Config:
    batch_size: int
    block_size: int
    n_embd: int
    learning_rate: float
    max_iters: int
    eval_interval: int
    eval_steps: int
    device: str


def main(c: Config):

    ### create dataset
    text = load_data()
    tokenizer = Tokenizer(text)
    c.n_embd = tokenizer.vocab_size

    data = torch.tensor(tokenizer.encode(text))

    n = int(0.9 * len(data))
    ds_train = Dataset(data[:n], bsz=c.batch_size, block_size=c.block_size)
    ds_val = Dataset(data[n:], bsz=c.batch_size, block_size=c.block_size)

    ### create model
    model = BigramLanguageModel(tokenizer.vocab_size, c.n_embd).to(c.device)

    ### train
    optim = torch.optim.AdamW(model.parameters(), lr=c.learning_rate)

    for steps in range(c.max_iters):
        # eval
        if steps % c.eval_interval == 0:
            loss_train = estimate_loss(model, ds_train, c.eval_steps)
            loss_val = estimate_loss(model, ds_val, c.eval_steps)
            print(
                f"steps: {steps}, loss_train: {loss_train:.5f}, loss_val: {loss_val:.5f}"
            )

        # forward
        xb, yb = ds_train.get_batch()
        xb, yb = xb.to(c.device), yb.to(c.device)
        _, loss = model(xb, yb)

        # backward
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=1000)
    print(tokenizer.decode(out.tolist()[0]))


if __name__ == "__main__":
    torch.manual_seed(1337)

    c = Config(
        batch_size=32,
        block_size=8,
        n_embd=32,
        learning_rate=1e-3,
        max_iters=1000,
        eval_interval=500,
        eval_steps=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    main(c)
