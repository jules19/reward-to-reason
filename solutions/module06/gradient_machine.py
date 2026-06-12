"""Module 6 solution — The Gradient Machine."""

import math
import random


class Dial:
    """A number that remembers how it was made.

    Build expressions out of Dials and every Dial records its parents and
    a little rule for passing blame backward. Call .backward() on the
    final result and every Dial in the expression learns its gradient:
    "if I had been slightly bigger, the result would have changed by this
    much."

    That one idea — blame, flowing backward through arithmetic — is the
    entire engine underneath every neural network on Earth.
    """

    def __init__(self, value, parents=(), backward_fn=None):
        self.value = value
        self.grad = 0.0
        self._parents = parents
        self._backward_fn = backward_fn

    def __add__(self, other):
        other = other if isinstance(other, Dial) else Dial(other)
        out = Dial(self.value + other.value, (self, other))

        def backward_fn():
            self.grad += out.grad
            other.grad += out.grad

        out._backward_fn = backward_fn
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Dial) else Dial(other)
        out = Dial(self.value * other.value, (self, other))

        def backward_fn():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad

        out._backward_fn = backward_fn
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Dial) else Dial(other)
        return self + (-other)

    __radd__ = __add__
    __rmul__ = __mul__

    def tanh(self):
        t = math.tanh(self.value)
        out = Dial(t, (self,))

        def backward_fn():
            self.grad += (1.0 - t * t) * out.grad

        out._backward_fn = backward_fn
        return out

    def relu(self):
        out = Dial(max(self.value, 0.0), (self,))

        def backward_fn():
            self.grad += (1.0 if self.value > 0.0 else 0.0) * out.grad

        out._backward_fn = backward_fn
        return out

    def backward(self):
        """Sort the graph so parents come first, then flow blame backward."""
        order, visited = [], set()

        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._parents:
                visit(parent)
            order.append(node)

        visit(self)
        self.grad = 1.0
        for node in reversed(order):
            if node._backward_fn is not None:
                node._backward_fn()


class Neuron:
    """A few Dials and a squash. The atom of every brain in Act III."""

    def __init__(self, n_inputs, rng):
        self.weights = [Dial(rng.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Dial(0.0)

    def __call__(self, inputs):
        total = self.bias
        for w, x in zip(self.weights, inputs):
            total = total + w * x
        return total.tanh()

    def parameters(self):
        return self.weights + [self.bias]


class TinyNet:
    """A hand-made neural network: layers of Neurons made of Dials."""

    def __init__(self, sizes, seed=0):
        rng = random.Random(seed)
        self.layers = []
        for n_in, n_out in zip(sizes, sizes[1:]):
            self.layers.append([Neuron(n_in, rng) for _ in range(n_out)])

    def __call__(self, inputs):
        values = [Dial(x) if not isinstance(x, Dial) else x for x in inputs]
        for layer in self.layers:
            values = [neuron(values) for neuron in layer]
        return values[0] if len(values) == 1 else values

    def parameters(self):
        return [p for layer in self.layers for n in layer for p in n.parameters()]


if __name__ == "__main__":
    # XOR: the function that broke single neurons in 1969.
    data = [((0, 0), -1.0), ((0, 1), 1.0), ((1, 0), 1.0), ((1, 1), -1.0)]
    net = TinyNet([2, 4, 1], seed=0)

    print("Teaching a hand-made brain XOR by turning dials downhill...\n")
    for epoch in range(401):
        loss = Dial(0.0)
        for inputs, target in data:
            prediction = net(inputs)
            error = prediction - target
            loss = loss + error * error
        for p in net.parameters():
            p.grad = 0.0
        loss.backward()
        for p in net.parameters():
            p.value -= 0.1 * p.grad
        if epoch % 100 == 0:
            print(f"  epoch {epoch:3d}: loss {loss.value:.4f}")

    print("\nfinal answers:")
    for inputs, target in data:
        print(f"  {inputs} -> {net(inputs).value:+.2f}   (truth {target:+.0f})")
    print("\nEvery weight found its place because blame flowed backward")
    print("through your arithmetic. You built the machine that learns.")
    print("\nTHE BUILT-IN CHIP RULE: your Dial works, so you have now earned")
    print("`from r2r.tensor import ...` — the same machine, vectorized.")
