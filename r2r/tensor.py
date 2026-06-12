"""tensor — the industrial-strength version of Module 6's Gradient Machine.

THE BUILT-IN CHIP RULE: you may not import this module until the Module 6
tests pass. You build the scalar version yourself first. Then — exactly as
Nand2Tetris lets you swap your hand-made chips for fast built-in ones —
you earn this numpy-powered drop-in.

It is the same machine you built, with one upgrade: instead of one number
per Dial, every node carries a whole numpy array, so thousands of
gradients flow in a single backward pass.
"""

import numpy as np


def _unbroadcast(grad, shape):
    """Sum `grad` back down to `shape` (undo numpy broadcasting)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    """An array that remembers how it was made, so blame can flow backward."""

    def __init__(self, data, parents=(), backward_fn=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._parents = parents
        self._backward_fn = backward_fn

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def backward_fn():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward_fn = backward_fn
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def backward_fn():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward_fn = backward_fn
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other))

        def backward_fn():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward_fn = backward_fn
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,))

        def backward_fn():
            self.grad += (self.data > 0.0) * out.grad

        out._backward_fn = backward_fn
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,))

        def backward_fn():
            self.grad += (1.0 - t * t) * out.grad

        out._backward_fn = backward_fn
        return out

    def log_softmax(self):
        """Row-wise log-softmax (for policy heads)."""
        shifted = self.data - self.data.max(axis=-1, keepdims=True)
        log_z = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        logp = shifted - log_z
        out = Tensor(logp, (self,))

        def backward_fn():
            softmax = np.exp(logp)
            self.grad += out.grad - softmax * out.grad.sum(axis=-1, keepdims=True)

        out._backward_fn = backward_fn
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,))

        def backward_fn():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward_fn = backward_fn
        return out

    def mean(self):
        n = self.data.size
        out = Tensor(self.data.mean(), (self,))

        def backward_fn():
            self.grad += np.ones_like(self.data) * (out.grad / n)

        out._backward_fn = backward_fn
        return out

    def backward(self):
        """Topologically sort the graph, then send blame back through it."""
        order, visited = [], set()

        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._parents:
                visit(parent)
            order.append(node)

        visit(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(order):
            if node._backward_fn is not None:
                node._backward_fn()


class Linear:
    """One layer of brain: y = x @ W + b."""

    def __init__(self, n_in, n_out, rng=None):
        rng = rng or np.random.default_rng()
        scale = np.sqrt(2.0 / n_in)
        self.W = Tensor(rng.normal(0.0, scale, size=(n_in, n_out)))
        self.b = Tensor(np.zeros(n_out))

    def __call__(self, x):
        return x.matmul(self.W) + self.b

    def parameters(self):
        return [self.W, self.b]


class MLP:
    """A multi-layer perceptron with ReLU between layers."""

    def __init__(self, sizes, rng=None):
        rng = rng or np.random.default_rng()
        self.layers = [Linear(a, b, rng) for a, b in zip(sizes, sizes[1:])]

    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        return self.layers[-1](x)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = list(parameters)
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
