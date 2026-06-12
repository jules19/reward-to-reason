# Reward2Reason

**Build a game engine that teaches itself — starting from a single bit
of reward.**

```
                                                        ┌─────────────┐
                                                        │   REASON    │  Module 10
                                                   ┌────┤ net + search│
                                                   │    │ + self-play │
                                       ┌───────────┴─┐  └─────────────┘
                                       │ IMAGINATION │   Module 9
                                  ┌────┤ tree search │
                                  │    └─────────────┘
                      ┌───────────┴─┐   Module 8
                      │  THE MIRROR │   self-play
                 ┌────┤  no teacher │
                 │    └─────────────┘
     ┌───────────┴─┐   Modules 6–7
     │  THE BRAIN  │   backprop, generalization
┌────┤  the wall   │   Modules 1–5
│    │  falls      │   reward → preference → doubt → place → foresight
│    └─────────────┘
REWARD            "a number that sometimes goes up"
```

In 1989, Nand2Tetris asked: what if a student built a whole computer,
from a single NAND gate to Tetris, with no magic anywhere in the stack?

This course asks the same question about intelligence:

> **What if you built a learning machine — from a single reward signal
> to an engine that masters games by playing against its own
> reflection — with no magic anywhere in the stack?**

## How it begins

You play Connect Four against a machine, and you lose. By the final
module that machine exists *because you typed it* — and yours is
stronger, because yours learns. In between, there is no week where you
"cover material." There is a tower, and you build every floor:

| # | Module | You build | The wall it breaks |
|---|--------|-----------|--------------------|
| 0 | [The Bet](modules/00-the-bet/) | — (you lose a board game to an empty file) | your certainty |
| 1 | [The Coin That Learns](modules/01-the-coin/) | preference from surprise — *the* nudge rule | "a number can't prefer" |
| 2 | [The Greedy Trap](modules/02-the-greedy-trap/) | exploration (ε-greedy) | lucky accidents harden into superstition |
| 3 | [A Place to Stand](modules/03-a-place-to-stand/) | a memory of places (Q-table, Monte Carlo) | a learner with no "here" |
| 4 | [Messages From the Future](modules/04-messages-from-the-future/) | Q-learning — guesses teaching guesses | learning only from finished stories |
| 5 | [The Wall](modules/05-the-wall/) | instruments, to measure your agent failing | — *(this module IS the wall)* |
| 6 | [The Gradient Machine](modules/06-the-gradient-machine/) | backpropagation, from bare floats | functions can't learn |
| 7 | [The Brain](modules/07-the-brain/) | deep Q-learning, generalization | memory can't cover the world |
| 8 | [The Mirror](modules/08-the-mirror/) | self-play — the infinite teacher | you can't outgrow your teacher |
| 9 | [Imagination](modules/09-imagination/) | minimax + Monte Carlo Tree Search | you can only judge what you've lived |
| 10 | [Reason](modules/10-reason/) | **AlphaZero-lite** — the loop that makes itself smarter | intuition and imagination were separate |
| 11 | [You Are Here](modules/11-you-are-here/) | — (no code; the view from the top) | — |

Every module ends with two things: something **running** that wasn't
possible yesterday, and a **measured failure** that makes tomorrow
necessary. You will never be taught a solution before you have
personally hit its problem.

## Quick start

```bash
pip install -e ".[dev]"
python -m r2r.play boss        # Module 0: lose the bet
pytest tests/test_module01.py  # red. start building.
python -m r2r.progress         # the tower, floor by floor, as you build it
```

Open [`modules/00-the-bet/README.md`](modules/00-the-bet/README.md) and
walk the modules in order — the order is the course. Each module is:
read the README, fill the TODOs in one Python file, make
`pytest tests/test_moduleNN.py` go green, then run the file itself and
watch the thing you built do something slightly unreasonable.

**Requirements:** Python 3.9+, numpy. No GPU, no frameworks, no cloud.
Everything trains in seconds-to-minutes on a laptop; the final Connect
Four engine enjoys an hour. ~1,500 lines of your own code, total.

## The rules of the house

1. **No magic imports.** The only learning library used in this course
   is the one you build in Module 6. (PyTorch would work fine. That's
   exactly why it's banned.)
2. **The Built-In Chip Rule** *(stolen with pride from Nand2Tetris)*:
   `r2r/` ships fast versions of several things you build — you may use
   one only after your own hand-made version passes the tests.
   Abstractions are earned here, never borrowed.
3. **Walls before doors.** When your agent fails, that failure is the
   curriculum. Measure it. The next module's idea will feel like a
   rescue, because it is one.
4. **The tests are the examiner** — and in Module 6, the examiner is
   literally calculus. Green means built. There is no partial credit
   and no essay section.

## What's in the box

```
modules/    the course — READMEs (the narrative) + stub files (your work)
tests/      one examiner per module; pytest tests/test_moduleNN.py
r2r/        provided infrastructure: worlds (bandits, mazes, cliffs,
            board games), the arena & Elo ladder, built-in reference
            agents, and the tensor library you'll earn in Module 6
solutions/  reference implementations — the course's own CI runs
            against these. The course works only if you don't peek;
            when truly stuck, read the module README again, then the
            test, then ask a human or an AI a QUESTION (not for the file)
```

## Why this course feels the way it feels

Because it optimizes for the only thing worth optimizing a course for:
the moment when a student stares at their own terminal and says *"wait —
I built that?"* It happens here roughly eleven times: a coin learns, a
map condenses out of wandering, a learner walks a cliff edge its
predecessor feared, a brain aces a world its table flunked, an agent
locked in a room with a mirror comes out unbeatable, and — the big one —
random static argues itself into reason on your screen, then beats you
at Connect Four.

The secret isn't the content. The secret is the *shape*: one primitive,
one narrative, walls before doors, every floor of the tower yours. For
the full design rationale (and how to build courses shaped like this in
other fields), see [DESIGN.md](DESIGN.md).

---

*Reward2Reason is an independent course inspired by the structure of
[Nand2Tetris](https://www.nand2tetris.org/) (Nisan & Schocken) and by
the canon of reinforcement learning it walks through — Watkins'
Q-learning, Tesauro's TD-Gammon, Mnih et al.'s DQN, and Silver et al.'s
AlphaZero, among others. Sutton & Barto's
[textbook](http://incompleteideas.net/book/the-book.html) is the
recommended companion for everything here.*
