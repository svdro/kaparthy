import torch
from torchpy.tensor import Tensor


def load_names():
    with open("./names.txt", "r") as f:
        return f.read().split("\n")


def test(a, b):
    c = a**b
    c.backward()
    print(c.__class__.__name__, "a: ", a.grad, "b: ", b.grad)
    print(a)


def main():
    a, b = 3.0, 10.0

    a_ = Tensor(a, requires_grad=True)
    b_ = Tensor(b, requires_grad=True)
    print("torchpy: ")
    test(a_, b_)

    a_ = torch.tensor(a, requires_grad=True)
    b_ = torch.tensor(b, requires_grad=True)
    print("\ntorch: ")
    test(a_, b_)


def get_training_examples(names: list[str], stoi: dict[str, int], context_len: int):
    X, Y = [], []

    for name in names:
        name += "."
        x = [stoi["."] for _ in range(context_len)]
        X.append(x[-3:])
        Y.append(stoi[name[0]])
        for ch1, ch2 in zip(name[:-1], name[1:]):
            x += [stoi[ch1]]
            X.append(x[-3:])
            Y.append(stoi[ch2])

    return X, Y


if __name__ == "__main__":
    names = load_names()
    vocab = sorted(set("".join(names) + "."))
    itos = {i: ch for i, ch in enumerate(vocab)}
    stoi = {ch: i for i, ch in itos.items()}

    print(len(names))
    print(vocab)
    print("itos: ", itos)
    print("stoi: ", stoi)
    print(names[:3])
    X, Y = get_training_examples(names, stoi, context_len=3)
    print(len(X), len(Y))

    # main()
