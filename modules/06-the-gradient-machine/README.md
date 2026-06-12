# Module 6 — The Gradient Machine

> *Wall broken this module: "a function can't learn — it has no way to
> know which of its knobs caused the mistake."*

## The blame problem

Module 5 ordered a memory that interpolates: a function with thousands
of internal knobs (weights), trainable from examples. Composing such a
function is easy — multiply, add, squash, repeat. The hard part is
this:

> The function's output is wrong by 0.3. There are 4,800 knobs. **Which
> knob do you turn, and which way?**

Turning knobs at random is evolution — it works, given a few million
years. The fast answer needs each knob to know its *responsibility* for
the error: "if this knob had been a hair larger, the error would have
been *this much* different." Calculus calls that a partial derivative.
We'll call it what it functions as: **blame**.

The machine you build today computes blame for every knob in any
arithmetic expression, automatically, in one backward sweep. It is — and
this is not a pedagogical simplification — the same machine inside
PyTorch, JAX, and every large model currently running. Theirs is faster.
It is not deeper.

## The whole idea, in one paragraph

A `Dial` is a number that **remembers how it was made**. When you write
`c = a * b`, the result carries a note: "I came from `a` times `b` —
and when blame reaches me, pass it on: `a`'s share is scaled by `b`'s
value, `b`'s by `a`'s." Addition's note is even simpler: "pass it
through unchanged." Build a whole expression and you've silently built a
graph of these notes. Call `.backward()` on the final result and blame
flows from the output to every input, each step applying one local note.
A thousand-layer network is just a long chain of notes. That's it.
That's backpropagation. The chain rule of calculus, performed by a data
structure.

## Build

Open [`gradient_machine.py`](gradient_machine.py). Five TODOs: `+`, `*`,
`tanh`, `relu`, `backward`. Two traps the tests are specifically lying
in wait for:

- **Blame accumulates.** A Dial used twice (`a * a`) collects blame from
  both uses — `+=`, never `=`.
- **Order matters.** Blame must flow output-to-inputs, so `backward()`
  first sorts the graph (parents before children) and then walks it in
  reverse.

Below your TODOs sit `Neuron` and `TinyNet`, prewritten — a neural
network made of nothing but your Dials. You build the atom; the brain
comes free.

Your examiner this module is calculus itself: the tests wiggle each
input by a millionth, measure how the output moves, and demand your
backward pass agree to five decimal places. You can't argue with the
examiner. The examiner is the definition of a derivative.

```bash
pytest tests/test_module06.py -q
python modules/06-the-gradient-machine/gradient_machine.py
```

## What you should see

The demo teaches your hand-made net **XOR** — the famous function whose
impossibility for single neurons froze this entire field for a decade
after 1969. Watch the loss collapse from 4.2 to 0.007 as four hundred
rounds of blame find every weight's proper place. The thaw that ended
that decade was exactly the machine you just built.

## The reveal

You built **reverse-mode automatic differentiation** — backpropagation.
Andrej Karpathy's `micrograd` is this same ~60 lines; PyTorch is this
plus a decade of engineering. Every gradient ever descended by a
trillion-parameter model flowed through the two rules you wrote in
`__add__` and `__mul__`. The atom of deep learning fits on an index
card, and yours is on it.

## The Built-In Chip Rule, invoked

Your Dial moves one float at a time through Python; training a real
brain that way would take all week. So, exactly as Nand2Tetris swaps
your hand-made chips for fast built-ins *after* you've built them, you
have now **earned**:

```python
from r2r.tensor import Tensor, MLP, Adam
```

Open [`r2r/tensor.py`](../../r2r/tensor.py) and actually read it — you
will recognize every line, because it is your Dial where each node
carries a numpy array instead of one float. Same notes, same backward
sweep, thousands of blames per pass. Nothing in that file is above your
pay grade anymore. That's what "earned" means.

## The crack in the wall

None. The wall from Module 5 still stands — we haven't touched it. What
you hold now is the tool the rescue specified: a function that can be
blamed into shape. Next module, you point it at the world that killed
the table.

→ [`modules/07-the-brain`](../07-the-brain/README.md)
