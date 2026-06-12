"""Module 6: The Gradient Machine.

The examiner here is calculus itself: every gradient your Dial reports
is checked against a numerical estimate (wiggle the input, measure the
output).
"""

import math

from r2r.course import load


def numeric_gradient(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def test_forward_values():
    Dial = load(6, "gradient_machine").Dial
    a, b = Dial(2.0), Dial(3.0)
    assert (a + b).value == 5.0
    assert (a * b).value == 6.0
    assert (a - b).value == -1.0
    assert abs(Dial(0.5).tanh().value - math.tanh(0.5)) < 1e-12
    assert Dial(-1.0).relu().value == 0.0
    assert Dial(1.5).relu().value == 1.5


def test_add_and_mul_gradients():
    Dial = load(6, "gradient_machine").Dial
    a, b = Dial(2.0), Dial(-3.0)
    out = a * b + b
    out.backward()
    assert abs(a.grad - (-3.0)) < 1e-9     # d(ab+b)/da = b
    assert abs(b.grad - 3.0) < 1e-9        # d(ab+b)/db = a + 1


def test_blame_accumulates_when_a_dial_is_reused():
    Dial = load(6, "gradient_machine").Dial
    a = Dial(3.0)
    out = a * a                            # d(a^2)/da = 2a = 6
    out.backward()
    assert abs(a.grad - 6.0) < 1e-9, \
        "a Dial used twice must collect blame from BOTH uses (use +=)"


def test_against_calculus_itself():
    Dial = load(6, "gradient_machine").Dial

    def expression(x):
        d = Dial(x)
        out = (d * 2.0 + 1.0).tanh() * d + d.relu() * 0.5
        out.backward()
        return out.value, d.grad

    for x in (-1.3, -0.2, 0.4, 1.7):
        value, grad = expression(x)
        expected = numeric_gradient(lambda v: expression(v)[0], x)
        assert abs(grad - expected) < 1e-5, \
            f"at x={x}: your gradient {grad}, calculus says {expected}"


def test_a_hand_made_brain_learns_xor():
    mod = load(6, "gradient_machine")
    data = [((0, 0), -1.0), ((0, 1), 1.0), ((1, 0), 1.0), ((1, 1), -1.0)]
    net = mod.TinyNet([2, 4, 1], seed=0)
    first_loss = None
    for epoch in range(250):
        loss = mod.Dial(0.0)
        for inputs, target in data:
            error = net(inputs) - target
            loss = loss + error * error
        if first_loss is None:
            first_loss = loss.value
        for p in net.parameters():
            p.grad = 0.0
        loss.backward()
        for p in net.parameters():
            p.value -= 0.1 * p.grad
    assert loss.value < 0.2 < first_loss, \
        "gradient descent through your Dial should crush the XOR loss"
