# Module 0 — The Bet

> *"Begin with the end in mind, then lose to it."*

## Setup (two minutes)

You need Python 3.9+ and nothing exotic. From the repository root:

```bash
pip install -e ".[dev]"
```

That's the whole installation. Check it worked:

```bash
pytest tests/test_module01.py -q
```

You should see **failures**. Good. Every red line is a thing you are
about to build. This course is finished when they are all green and the
machine described below exists because you typed it into existence.

## The fight

Now play this:

```bash
python -m r2r.play boss
```

You are playing Connect Four against a machine. Type a column number to
drop a piece. Take your time. Try to win.

You will probably lose. If you win, congratulations — now run it again;
it won't happen twice.

## What just beat you

Here is the strange part. The thing that beat you:

- contains **no Connect Four strategy**. Nobody programmed an opening
  book, a tactic, a heuristic, a rule of thumb. Search the source — you
  will find none.
- was **never shown a single game** played by anyone who knows what
  they're doing.
- doesn't even particularly "know" it is playing Connect Four.

And yet it just beat you, a creature with a hundred billion neurons and
a childhood full of board games.

By the end of this course you will have built every meaningful line of
that machine yourself — and a strictly stronger one: yours will *learn*,
improving itself by playing against its own reflection, starting from
literally random flailing.

## The bet

Here is the wager this course makes, stated as plainly as possible:

> **Everything that machine has — preference, memory, judgement,
> intuition, foresight — can be grown from a single primitive: a number
> that sometimes goes up.**

That number is called **reward**. It is the NAND gate of intelligence.

You will start with a learner so small it is almost insulting — two
numbers and one nudge rule — and you will not be handed anything else
for the rest of the course. Every new capability must be *forced into
existence by a problem the previous capability cannot solve*. You will
hit each wall personally, measure it, and then build the exact tool that
breaks it.

## The route

| Act | Modules | You build | The wall it breaks |
|-----|---------|-----------|--------------------|
| **I. The Spark** | 1–4 | A learner: preference, doubt, place-memory, foresight | "Behaviour can't come from a number" |
| **II. The Wall** | 5–7 | A brain: backprop, generalization | "Memory can't cover the world" |
| **III. The Mind** | 8–10 | A mind: self-teaching, imagination, reason | "Learning can't outgrow its teacher" |

Twelve modules. Each one ends with something *running* — and with a
specific, demonstrated failure that makes the next module necessary.

One rule, borrowed from Nand2Tetris, called the **Built-In Chip Rule**:
the `r2r/` package contains polished versions of several things you will
build (tree search, a tensor library). You may use a built-in component
only *after* your own hand-made version of it passes the tests. You earn
abstractions here; you don't borrow them.

## Remember this feeling

You just lost a board game to an empty file. Hold on to how implausible
that feels. In ten modules, the implausible thing will be sitting in
your working tree, and you will know — not believe, *know* — that there
was no magic involved at any step.

The magic has to go somewhere, though. Watch carefully where it goes.

→ Proceed to [`modules/01-the-coin`](../01-the-coin/README.md).
