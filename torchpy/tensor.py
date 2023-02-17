from __future__ import annotations
from typing import Optional, Callable

from .autograd import (
    backward,
    grad_fn,
    MulBackward,
    AddBackward,
    SubBackward,
    PowBackward,
    ReluBackward,
    DivBackward,
)


def mul(a: Tensor, b: Tensor) -> Tensor:
    out_requires_grad = a.requires_grad or b.requires_grad
    out_grad_fn = MulBackward(a, b) if out_requires_grad else None
    return Tensor(a.data * b.data, requires_grad=out_requires_grad, grad_fn=out_grad_fn)


def add(a: Tensor, b: Tensor) -> Tensor:
    out_requires_grad = a.requires_grad or b.requires_grad
    out_grad_fn = AddBackward(a, b) if out_requires_grad else None
    return Tensor(a.data + b.data, requires_grad=out_requires_grad, grad_fn=out_grad_fn)


def div(a: Tensor, b: Tensor):
    out_requires_grad = a.requires_grad or b.requires_grad
    out_grad_fn = DivBackward(a, b) if out_requires_grad else None
    return Tensor(a.data / b.data, requires_grad=out_requires_grad, grad_fn=out_grad_fn)


def sub(a: Tensor, b: Tensor) -> Tensor:
    out_requires_grad = a.requires_grad or b.requires_grad
    out_grad_fn = SubBackward(a, b) if out_requires_grad else None
    return Tensor(a.data - b.data, requires_grad=out_requires_grad, grad_fn=out_grad_fn)


def pow(a: Tensor, b: Tensor) -> Tensor:
    out_requires_grad = a.requires_grad or b.requires_grad
    out_grad_fn = PowBackward(a, b) if out_requires_grad else None

    return Tensor(
        a.data**b.data, requires_grad=out_requires_grad, grad_fn=out_grad_fn
    )


def relu(a: Tensor):
    out_grad_fn = ReluBackward(a) if a.requires_grad else None
    return Tensor(max(a.data, 0), requires_grad=a.requires_grad, grad_fn=out_grad_fn)


class Tensor:
    def __init__(
        self,
        data: int | float,
        requires_grad: bool = False,
        grad_fn: Optional[grad_fn] = None,
    ):

        self.requires_grad = requires_grad
        self.data = float(data)

        self.grad = None
        self.grad_fn: Optional[grad_fn] = grad_fn

    def backward(self, gradient: Optional[Tensor] = None):
        self.grad = gradient or Tensor(1.0)
        assert self.grad_fn is not None
        backward(self.grad_fn, self.grad or Tensor(1.0))

    def relu(self):
        return relu(self)

    def __mul__(self, other: int | float | Tensor):
        return mul(self, other if isinstance(other, Tensor) else Tensor(other))

    def __add__(self, other: int | float | Tensor):
        return add(self, other if isinstance(other, Tensor) else Tensor(other))

    def __sub__(self, other: int | float | Tensor):
        return sub(self, other if isinstance(other, Tensor) else Tensor(other))

    def __truediv__(self, other: int | float | Tensor):
        return div(self, other if isinstance(other, Tensor) else Tensor(other))

    def __neg__(self):
        return mul(self, Tensor(-1))

    def __pow__(self, other: int | float | Tensor):
        print("pow")
        return pow(self, other if isinstance(other, Tensor) else Tensor(other))

    def __rmul__(self, other: int | float | Tensor):
        return self.__mul__(other)

    def __rtruediv__(self, other: int | float | Tensor):
        return div(other if isinstance(other, Tensor) else Tensor(other), self)

    def __radd__(self, other: int | float | Tensor):
        return self.__add__(other)

    def __rpow__(self, other: int | float | Tensor):
        print("rpow")
        return pow(other if isinstance(other, Tensor) else Tensor(other), self)

    def __repr__(self):
        if self.grad_fn:
            return f"tensor({self.data}, grad_fn={self.grad_fn})"

        if self.requires_grad:
            return f"tensor({self.data}, requires_grad={self.requires_grad})"

        return f"tensor({self.data})"
