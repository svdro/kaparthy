from __future__ import annotations
import math

from . import tensor
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .tensor import Tensor


class grad_fn:
    def __init__(self, next_functions: Optional[list[grad_fn]] = None):
        self.next_functions = next_functions

    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, grad: Tensor) -> Optional[tuple[Tensor, ...]]:
        raise NotImplementedError


class AccumulateGrad(grad_fn):
    def __init__(self, a: Tensor):
        super(AccumulateGrad, self).__init__()
        self.a = a

    def __call__(self, grad: Tensor) -> Optional[tuple[Tensor, ...]]:
        if self.a.grad is None:
            self.a.grad = tensor.Tensor(0.0)

        self.a.grad += grad


class AddBackward(grad_fn):
    def __init__(self, a: Tensor, b: Tensor):
        super(AddBackward, self).__init__(get_next_functions(a, b))
        self.a, self.b = a, b

    def __call__(self, grad: Tensor):
        grad_a = grad * 1 if self.a.requires_grad else None
        grad_b = grad * 1 if self.b.requires_grad else None
        return tuple(grad for grad in (grad_a, grad_b) if grad is not None)


class SubBackward(grad_fn):
    def __init__(self, a: Tensor, b: Tensor):
        super(SubBackward, self).__init__(get_next_functions(a, b))
        self.a, self.b = a, b

    def __call__(self, grad: Tensor):
        grad_a = grad * 1 if self.a.requires_grad else None
        grad_b = grad * -1 if self.b.requires_grad else None
        return tuple(grad for grad in (grad_a, grad_b) if grad is not None)


class MulBackward(grad_fn):
    def __init__(self, a: Tensor, b: Tensor):
        super(MulBackward, self).__init__(get_next_functions(a, b))
        self.a, self.b = a, b

    def __call__(self, grad: Tensor) -> Optional[tuple[Tensor, ...]]:
        grad_a = grad * tensor.Tensor(self.b.data) if self.a.requires_grad else None
        grad_b = grad * tensor.Tensor(self.a.data) if self.b.requires_grad else None
        return tuple(grad for grad in (grad_a, grad_b) if grad is not None)


class DivBackward(grad_fn):
    def __init__(self, a: Tensor, b: Tensor):
        super(DivBackward, self).__init__(get_next_functions(a, b))
        self.a, self.b = a, b

    def __call__(self, grad: Tensor) -> Optional[tuple[Tensor, ...]]:
        grad_a = tensor.Tensor(self.b.data**-1)
        grad_b = tensor.Tensor(-1 * self.a.data * self.b.data**-2)
        grad_a = grad * grad_a if self.a.requires_grad else None
        grad_b = grad * grad_b if self.b.requires_grad else None
        return tuple(grad for grad in (grad_a, grad_b) if grad is not None)


class PowBackward(grad_fn):
    def __init__(self, a: Tensor, b: Tensor):
        super(PowBackward, self).__init__(get_next_functions(a, b))
        self.a, self.b = a, b

    def __call__(self, grad: Tensor) -> Optional[tuple[Tensor, ...]]:
        grad_a = tensor.Tensor(self.b.data * self.a.data ** (self.b.data - 1))
        grad_b = tensor.Tensor(self.a.data**self.b.data * math.log(self.a.data))
        grad_a = grad * grad_a if self.a.requires_grad else None
        grad_b = grad * grad_b if self.b.requires_grad else None
        return tuple(grad for grad in (grad_a, grad_b) if grad is not None)


class ReluBackward(grad_fn):
    def __init__(self, a: Tensor):
        super(ReluBackward, self).__init__(get_next_functions(a))
        self.a = a

    def __call__(self, grad: Tensor):
        grad_a = 1 if self.a.data > 0 else 0
        grad_a = grad * grad_a if self.a.requires_grad else None
        return tuple(grad for grad in [grad_a] if grad is not None)


def get_next_functions(*args: Tensor) -> Optional[list[grad_fn]]:
    next_functions = [get_next_function(t) for t in args]
    return [nf for nf in next_functions if nf is not None] or None


def get_next_function(t: Tensor) -> Optional[grad_fn]:
    return t.grad_fn if t.grad_fn else AccumulateGrad(t) if t.requires_grad else None


def backward(grad_fn: grad_fn, gradient: Tensor):
    gradients = grad_fn(gradient)
    next_functions = grad_fn.next_functions

    if gradients is None or next_functions is None:
        return

    for gradient, grad_fn in zip(gradients, next_functions):
        backward(grad_fn, gradient)
