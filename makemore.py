import argparse
import os
import time
import torch
import torch.nn.functional as F
import re
import sys

from dataclasses import dataclass
from torch import nn, Tensor, optim
from typing import Iterator, Optional, Tuple, Sized

# -----------------------------------------------------------------------------
# Config


@dataclass
class Config:
    # Model
    vocab_size: int
    seq_length: int
    n_layers: int = 2
    embed_dim: int = 8
    hidden_dim: int = 100

    # Training
    learning_rate: float = 1e-1
    lr_decay: float = 1.0
    n_epochs: int = 10
    batch_size: int = 32

    # Dataset
    num_workers: int = 0
    persistent_workers: bool = False

    # Indices
    _IGNORE_IDX: int = -1
    _PAD_IDX: int = 0
    _EOS_IDX: int = 1


# -----------------------------------------------------------------------------
# Model Base Class


class Model(nn.Module):
    """Model base class"""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: tensor, torch.int64, shape[batch_size, max_seq_len]

        Returns:
            x: tensor, torch.float32, shape[batch_size * max_word_length, vocab_size]
        """
        raise NotImplementedError

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# MLP Model


class MLP(Model):
    _name = "MLP"

    def __init__(self, c: Config):
        """Essentially an embedding layer followed by "n_layers" dense layers."""
        super().__init__()
        self._pad_idx = c._PAD_IDX
        self.seq_length = c.seq_length
        self.vocab_size = c.vocab_size
        # c.vocab_size + 1 to account for PAD_IDX
        self.embeddings = nn.Embedding(c.vocab_size + 1, c.embed_dim)

        d_in = lambda i: c.hidden_dim if i != 0 else c.embed_dim * c.seq_length
        d_out = lambda i: c.hidden_dim if i != c.n_layers - 1 else c.vocab_size
        self.linears = nn.ModuleList(
            [nn.Linear(d_in(i), d_out(i)) for i in range(c.n_layers)]
        )

    def preprocess_batch(self, idx: Tensor) -> Tensor:
        """
        Args:
            idx:     torch.int64  (batch_size, max_seq_length)

        Returns:
            idx:     torch.float32 (batch_size * max_seq_length, seq_len * embed_dim)

        Preprocesses a model inputs to fit into mlp model architecture and fetches
        character embeddings. The effective batch size is the input's batch size
        multiplied by max_seq_length.

        e.g:
            x                        -> y
            [[.reinier......], ... ] -> [[reinier.00000], ... ]

            [[.------------],           [r
              [r.-----------],           e
              [er.----------],           i
              [ier.---------],           n
              [nier.--------],       ->  i
              [inier.-------],           e
              [einier.------],           r
              [reinier.-----],           .
              ....
              [...........re]...]       0]

        """
        embs = []
        for _ in range(self.seq_length):
            tok_emb = self.embeddings(idx)
            embs.append(tok_emb)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size  # PAD_IDX

        x = torch.cat(embs, dim=-1)
        return x.view(-1, x.size(-1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: tensor, torch.int64, shape[batch_size, max_seq_len]

        Returns:
            x: tensor, torch.float32, shape[batch_size * max_word_length, vocab_size]
        """
        # x = self.embeddings(x).view(x.size(0), -1)
        x = self.preprocess_batch(x)
        for i, l in enumerate(self.linears):
            x = l(x)
            if i < len(self.linears) - 1:
                x = torch.tanh(x)

            # x = torch.tanh(l(x)) if i < len(self.linears) - 1 else l(x)
        return x

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            x: tensor, torch.float32, shape[batch_size * max_word_length, vocab_size]
            y: tensor, torch.int64,   shape[batch_size, max_seq_len]

        Returns:
            loss: tensor, torch.float32, shape[]
        """
        return F.cross_entropy(logits, targets.view(-1), ignore_index=-1)


# -----------------------------------------------------------------------------
# data utils

from torch.utils.data import Dataset, RandomSampler
from torch.utils.data.dataloader import DataLoader


class CharDataset(Dataset):
    def __init__(self, words: list[str], chars: list[str], max_seq_length: int):
        self.words = words
        self.chars = ["."] + chars
        self.max_seq_length = max_seq_length

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def __len__(self):
        return len(self.words)

    def contains(self, word: str):
        return word in self.words

    @property
    def vocab_size(self):
        return len(self.chars)

    def get_output_length(self):
        return self.max_seq_length + 1  # <start> token

    def encode(self, word: str):
        return torch.tensor([self.stoi[w] for w in word], dtype=torch.long)

    def decode(self, x: Tensor):
        """x must be a 1d tensor"""
        word = "".join(self.itos[int(i.item())] for i in x)
        return re.findall(r"\w+", word)[0]

    def __getitem__(self, idx):
        ix = self.encode(self.words[idx])
        x = torch.zeros(self.max_seq_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_seq_length + 1, dtype=torch.long)

        x[1 : 1 + len(ix)] = ix
        y[: len(ix)] = ix
        y[len(ix) + 1 :] = -1  # -1 will mask the loss at inactive locations
        return x, y


class InfiniteDataLoader:
    def __init__(self, dataset: Sized, bsz: int, **kwargs):
        assert isinstance(dataset, Dataset)

        # ts = RandomSampler(dataset, replacement=False)
        ts = RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=ts, batch_size=bsz, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch


class EpochLoader:
    def __init__(self, dataset: Sized, bsz: int, **kwargs):
        assert isinstance(dataset, Dataset)

        ts = RandomSampler(dataset, replacement=False)
        self.train_loader = DataLoader(dataset, sampler=ts, batch_size=bsz, **kwargs)
        self.data_iter = iter(self.train_loader)

    def __len__(self):
        assert isinstance(self.train_loader.dataset, Sized)
        return len(self.train_loader.dataset)

    def yield_batches(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for batch in self.train_loader:
            yield batch


def create_datasets(input_file: str, verbose: bool = True):
    with open(input_file, "r") as f:
        words = f.read().splitlines()
        words = [w.strip() for w in words if w]
        chars = sorted(set("".join(words)))
        max_word_length = max(len(w) for w in words)

        log_msg = "\n" + "-" * 60 + "\n"
        log_msg += f"DATASET:\n path: {input_file}\n n examples: {len(words)}\n"
        log_msg += f" max word len: {max_word_length}\n "
        log_msg += f"vocab size: {len(chars)} \n vocab: {''.join(chars)}\n"
        log_msg += "-" * 60 + "\n"
        if verbose:
            print(log_msg)

        n_test = min(int(len(words) * 0.1), 1000)
        inds = torch.randperm(len(words)).tolist()
        train_words = [words[i] for i in inds[:-n_test]]
        test_words = [words[i] for i in inds[-n_test:]]
        trainDataset = CharDataset(train_words, chars, max_word_length)
        testDataset = CharDataset(test_words, chars, max_word_length)
        return trainDataset, testDataset


# -----------------------------------------------------------------------------
# train uitls


@torch.no_grad()
def eval(testLoader: EpochLoader, model: Model, c: Config) -> float:
    model.eval()

    losses = []
    for x, y in testLoader.yield_batches():
        logits = model(x)
        loss = model.compute_loss(logits, y)
        losses.append(loss.item())

    return sum(losses) / len(losses)


def train_step(x: Tensor, y: Tensor, model: Model, optimizer: optim.Optimizer) -> float:
    model.train()

    logits = model(x)
    loss = model.compute_loss(logits, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(
    trainLoader: EpochLoader,
    testLoader: EpochLoader,
    model: Model,
    c: Config,
    model_path: str,
    sample_interval: int = 0,
    verbose: bool = True,
):

    optimizer = optim.Adam(model.parameters(), lr=c.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=c.lr_decay)

    n_batches = len(trainLoader) // c.batch_size
    training_history = {"losses": [], "learning_rates": []}

    lr = c.learning_rate
    best_loss = eval(testLoader, model, c)

    for e in range(1, c.n_epochs + 1):
        t = time.time()
        lr = scheduler.get_last_lr()[0]

        for x, y in trainLoader.yield_batches():
            loss = train_step(x, y, model, optimizer)

            training_history["losses"].append(loss)
            training_history["learning_rates"].append(lr)

        avg_loss = sum(training_history["losses"][-n_batches:]) / n_batches
        test_loss = eval(testLoader, model, c)

        log_msg = f"({e}) -> {time.time()-t:.2f} sec | lr: {lr:.5f} |  avg loss: {avg_loss:.3f}"
        log_msg += f" | test loss: {test_loss:.3f}"

        if best_loss is None or test_loss < best_loss:
            torch.save(model.state_dict(), model_path)
            best_loss = test_loss
            log_msg += f" || saved model @ {model_path}"

        if verbose:
            print(log_msg)

            if sample_interval and e % sample_interval == 0:
                dsTrain = trainLoader.train_loader.dataset
                dsTest = testLoader.train_loader.dataset
                # for typechecker
                assert isinstance(dsTest, CharDataset)
                assert isinstance(dsTrain, CharDataset)

                words = rainbow_sample(model, dsTrain.stoi, top_k=1)
                words = [color_code(w, dsTrain, dsTest) for w in words]
                words = [
                    f"{w:32s}" if (i + 1) % 14 != 0 else f"{w}\n"
                    for i, w in enumerate(words)
                ]
                print("".join(words), "\n")
        scheduler.step()

    return training_history


# -----------------------------------------------------------------------------
# sampling


def generate_batch(
    model: nn.Module,
    idx: Tensor,
    seq_len: int,
    max_new_tokens: int = 10,
    top_k: int = 1,
) -> Tensor:
    """
    idx: (bsz, starting_seq_len)
    Returns: (bsz, starting_seq_len + max_new_tokens)
    """
    model.eval()

    for _ in range(max_new_tokens):
        logits = model(idx[:, -seq_len:])
        logits = logits.view(idx.size(0), -1, logits.size(-1))[:, -1, :]

        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("Inf")
        probas = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probas, num_samples=1)

        idx = torch.cat((idx, next_idx), dim=-1)

    return idx


def rainbow_sample(
    model: nn.Module,
    stoi: dict[str, int],
    top_k: int = 1,
) -> list[str]:
    """
    Samples each letter of the alphabet, formats the resulting list of names,
    and prints to stdout.
    """
    alphabet = torch.tensor(list(stoi.values()))[1:].unsqueeze(-1)
    bsz = alphabet.size(0)

    x = torch.zeros(bsz, 1, dtype=torch.long)
    x = torch.cat((x, alphabet), dim=-1)
    x = generate_batch(model, x, seq_len, max_new_tokens=10, top_k=top_k)

    return [dsTrain.decode(x_) for x_ in x]


def color_code(w: str, dsTrain: CharDataset, dsTest: CharDataset):
    r, y, b = "\033[0;31m", "\033[0;32m", "\033[0;34m"
    reset = "\033[0m"
    c = r if dsTrain.contains(w) else y if dsTest.contains(w) else b
    return c + w + reset


def sample(
    model: nn.Module,
    dsTrain: CharDataset,
    dsTest: CharDataset,
    prompt: Optional[str] = None,
    batch_size: int = 16,
    top_k: int = 1,
) -> list[str]:

    x = torch.zeros((batch_size, 1), dtype=torch.long)

    if prompt is not None:
        p = torch.tensor([dsTrain.stoi[ch] for ch in prompt]).unsqueeze(0)
        p = torch.ones((batch_size, p.size(1)), dtype=torch.long) * p
        x = torch.cat((x, p), dim=-1)

    x = generate_batch(model, x, seq_len, 10, top_k=10)
    words = [dsTrain.decode(x_) for x_ in x]
    return [color_code(w, dsTrain, dsTest) for w in words]


# -----------------------------------------------------------------------------
# main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make more tutorial")

    # input-output-general
    parser.add_argument("--input-file", "-", type=str, default="names.txt")
    parser.add_argument("--out-dir", "-o", type=str, default="out")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--seed", default=3407)
    parser.add_argument("--num-workers", type=int, default=1)

    # Sampling
    parser.add_argument("--sample", "-s", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)

    # model parameters
    parser.add_argument("--seq-length", type=int, default=0)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)

    # traing parameters
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--learning-rate", "-l", type=float, default=5e-4)
    parser.add_argument("--lr_decay", "-d", type=float, default=1.0)

    args = parser.parse_args()

    # seeds & stuff
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init datasets
    dsTrain, dsTest = create_datasets(args.input_file, verbose=True)

    # init config classes
    seq_len = args.seq_length if args.seq_length > 0 else dsTrain.get_output_length()
    config = Config(
        # Model
        vocab_size=dsTrain.vocab_size,
        # seq_length=args.seq_length,
        seq_length=seq_len,
        n_layers=args.n_layers,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        # Training
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        # Dataset
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    print("CONFIG: ")
    print(config, "\n")

    # build model
    model = MLP(config)
    model_path = os.path.join(args.out_dir, f"{model._name}_model.pt")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.resume or args.sample:
        model.load_state_dict(torch.load(model_path))
        print(f"resuming from existing model: {model_path}")

    # sample
    if args.sample:
        # rainbow_sample(model, dsTrain.stoi, top_k=5)
        words = sample(model, dsTrain, dsTest, args.prompt)
        for w in words:
            print(w)

        sys.exit(1)

    # train
    trainLoader = EpochLoader(
        dsTrain,
        bsz=config.batch_size,
        pin_memory=False,
        num_workers=config.num_workers,
        persistent_workers=config.persistent_workers,
        drop_last=False,
    )

    testLoader = EpochLoader(
        dsTest,
        bsz=2 * config.batch_size,
        pin_memory=False,
        num_workers=config.num_workers,
        persistent_workers=config.persistent_workers,
        drop_last=False,
    )

    train(trainLoader, testLoader, model, config, model_path, 2, True)
