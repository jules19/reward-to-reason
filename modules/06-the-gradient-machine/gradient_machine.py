"""Module 6 — The Gradient Machine.

Build the machine that learns: a number that remembers how it was made,
so that blame can flow backward through arithmetic.

Read README.md first.
Check your work:   pytest tests/test_module06.py
See it live:       python modules/06-the-gradient-machine/gradient_machine.py
"""

import math
import random


class Dial:
    """A number that remembers how it was made.

    Build expressions out of Dials and every Dial records its parents and
    a little rule for passing blame backward. Call .backward() on the
    final result and every Dial in the expression learns its gradient:
    "if I had been slightly bigger, the result would have changed by this
    much."

    That one idea is the entire engine underneath every neural network
    on Earth. You are about to build it in ~60 lines.
    """

    def __init__(self, value, parents=(), backward_fn=None):
        self.value = value
        self.grad = 0.0
        self._parents = parents
        self._backward_fn = backward_fn

    def __add__(self, other):
        """out = self + other.

        Blame rule for +: the output's blame flows to BOTH parents,
        unchanged. (If out wanted to be 1 bigger, either parent being 1
        bigger would have done it.)
        """
        other = other if isinstance(other, Dial) else Dial(other)
        # TODO 1:
        #   out = Dial(self.value + other.value, (self, other))
        #   define backward_fn():  self.grad += out.grad
        #                          other.grad += out.grad
        #   attach it (out._backward_fn = backward_fn) and return out.
        # NOTE the +=, never = : a Dial used twice collects blame twice.
        raise NotImplementedError("TODO 1")

    def __mul__(self, other):
        """out = self * other.

        Blame rule for *: each parent's blame is the output's blame
        scaled by the OTHER parent's value.
        """
        other = other if isinstance(other, Dial) else Dial(other)
        # TODO 2
        raise NotImplementedError("TODO 2")

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Dial) else Dial(other)
        return self + (-other)

    __radd__ = __add__
    __rmul__ = __mul__

    def tanh(self):
        """The squash. Blame rule: scale by (1 - tanh^2)."""
        # TODO 3: t = math.tanh(self.value); out = Dial(t, (self,))
        #         backward_fn: self.grad += (1 - t*t) * out.grad
        raise NotImplementedError("TODO 3")

    def relu(self):
        """max(0, x). Blame rule: pass blame through only if value > 0."""
        # TODO 4
        raise NotImplementedError("TODO 4")

    def backward(self):
        """Make every Dial in this expression learn its gradient.

        Two steps:
          1. Topological sort: list every node so that parents come
             before children (depth-first from here, append after
             visiting parents).
          2. Set self.grad = 1.0 (the result is fully to blame for
             itself), then walk the list IN REVERSE calling each node's
             _backward_fn (if it has one).
        """
        # TODO 5
        raise NotImplementedError("TODO 5")


# ---------------------------------------------------------------------------
# Everything below is prewritten and uses only your Dial. No TODOs.
# ---------------------------------------------------------------------------


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
