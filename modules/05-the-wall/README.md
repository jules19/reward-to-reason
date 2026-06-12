# Module 5 — The Wall

> *Nothing is broken this module. This module is the wall.*

## A different kind of module

No new algorithm. No new agent. This module exists because of a
discipline this course stole from Nand2Tetris and takes deadly
seriously: **you don't get the solution until you've been properly hurt
by the problem.** Most courses mention limitations in a sentence on the
way to the fix ("of course, tabular methods don't scale...") and the
student nods along, learning nothing. You are going to *measure* the
failure, with instruments you build yourself, until the next module's
idea feels less like a topic and more like a rescue.

## The world that doesn't repeat

`r2r.envs.OpenWorld` is an empty 16×16 room. No walls, no tricks. The
catch: **every episode, the goal moves.** A situation is now the pair
*(where I am, where the goal is)* — and there are 65,280 of them.

Your Module 4 agent is not in any way "wrong" here. Q-learning's
convergence guarantees hold. Every update is sound. Run enough episodes
— enough to visit all 65,280 situations many times each — and the table
*would* learn this world. Nobody has that many lifetimes. That's the
point. The wall isn't a bug; it's arithmetic.

## Build the instruments

Open [`the_wall.py`](the_wall.py). Three measuring tools:

- `count_states(env)` — how big is this world, really?
- `coverage(q_table, env)` — what fraction of it has the agent *ever seen*?
- `success_rate(value_fn, env)` — the honest exam: drop the greedy agent
  into **fresh** situations and count how often it reaches the goal.

That third one is the only exam that has ever mattered, in this course or
anywhere: *performance on situations you didn't train on.* Hold onto it;
you'll use it for the rest of the course.

```bash
pytest tests/test_module05.py -q
python modules/05-the-wall/the_wall.py
```

## The autopsy

The demo trains your Q-learner for 4,000 episodes — forty times what the
maze needed — then reads the instruments. Typical verdict:

```
situations ever visited : 84%
success on fresh exams  : 18%
```

Look at those two numbers together. **It has stood in 84% of all
possible situations and still fails four exams in five.** Standing
somewhere once isn't knowing it — a visited state whose value was nudged
twice by a wandering random walk knows almost nothing. The agent isn't
short of experience. It's short of a way for experience to *add up*.

And here is the sentence to underline, the actual diagnosis: in a
Q-table, **every entry is a private universe.** The agent may know
perfectly how to get from (3,4) to the goal at (9,9) — and that
knowledge contributes *exactly nothing* to the almost identical question
of getting there from (3,5). The table has no way to notice that two
situations are similar, because the table has no concept of *similar*.
It can only ask "have I been in exactly this situation before?" — and in
any world worth inhabiting, the answer is almost always no.

Scale the disaster: tic-tac-toe, ~5,500 states — a table laughs.
Connect Four, ~10¹³ — every table on Earth is dead. Chess, ~10⁴⁴ —
there are not enough atoms in the planet to build the table. The wall
isn't ahead of us on the road to interesting worlds. It's behind us.
Tables were never going to make it.

## What the rescue must look like

Don't turn the page yet. Specify the tool first, like an engineer:

We need a memory that, told "(3,4)→(9,9) is worth 0.81", gets *(3,5)→(9,9)
roughly right for free*. A memory where nearby questions share their
answers. Not a lookup — a **function**: four numbers in, judgement out,
with the property that small changes in the question produce small
changes in the answer. Then learning a million situations from a
thousand examples stops being magic and becomes interpolation.

Machines like that exist. The catch: a function that *learns* needs a
way to know how each of its internal knobs contributed to each mistake —
blame, flowing backward through arithmetic. Nobody hands that to you
here. Next module you build it, from bare Python floats.

→ [`modules/06-the-gradient-machine`](../06-the-gradient-machine/README.md)
