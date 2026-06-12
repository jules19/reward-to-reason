# Module 9 — Imagination

> *Wall broken this module: "the only way to evaluate a move is to have
> learned about it."*

## A road with no learning on it

Every module so far has been about learning — distilling the past into
judgement. This module builds strength from the opposite direction, and
it is the only module in the course with **zero learning in it**: no
values, no nudges, no memory between moves. Pure thought, at decision
time.

The trick: you hold the *rules* of the game, and rules are a simulator.
Before committing to a move, **imagine futures** — play them out in your
head, see how they end, and let the endings vote. Strength without
experience. It feels like it shouldn't count. It counts.

## Build one: exhaustive imagination

`Minimax` imagines *everything*. The definition is two lines of
recursion, and it's worth deriving aloud: a finished game's value is its
result. An unfinished one? *Whatever value the mover can force* — each
legal move leads to a position whose value you (recursively) know, and
the mover picks the best for their side, `max(player * value)`. That's
the whole algorithm. Add a cache — positions recur constantly — and
tic-tac-toe collapses: **4,520 positions, mapped in well under a
second.** The game you played as a child is, completely and forever,
solved on your screen, by twenty lines.

Then the cold shower, and it's the same wall as Module 5 wearing a new
mask: Connect Four, ~10¹³ positions. The recursion that ate tic-tac-toe
in milliseconds would gnaw on Connect Four for centuries. Exhaustive
imagination dies in any game worth playing — **combinatorial explosion**
kills the thinker the way state explosion killed the table.

## Build two: imagination on a budget

So imagine *selectively*. `MCTS` — Monte Carlo Tree Search — runs a
fixed number of "dreams" per move, each in four phases:

1. **Select** — descend through moves you've dreamt about before,
   preferring promising ones but sometimes doubting (read on);
2. **Expand** — step once into a move you've never imagined;
3. **Rollout** — finish the game *at random* (astonishing but true:
   thousands of stupid endings, averaged, point surprisingly well);
4. **Backpropagate** — every move on the path records the result.

After the budget is spent, play the move you kept wanting to dream
about.

Now look at the selection rule you're asked to implement:

```
score(move) = average_result + c * sqrt( log(parent_visits) / visits )
```

Greed plus a curiosity bonus that *grows for the under-tried* — you have
seen this exact dilemma before. It is Module 2's casino. **Every node of
the search tree is a tiny casino**, balancing exploiting good lines
against exploring doubtful ones, and the rule (UCB) is the principled
big sibling of your ε. The course's first lesson, reborn inside a
dream. (The field made the same journey: bandits 1950s, UCT 2006.)

## Build

Open [`imagination.py`](imagination.py): minimax with cache, then the
four-phase loop.

```bash
pytest tests/test_module09.py -q
python modules/09-imagination/imagination.py
```

## What you should see

Minimax solves tic-tac-toe instantly, then the demo shows MCTS playing
Connect Four — a game it could no more enumerate than you could — at
crushing strength against random play with a budget of just 400 dreams
per move, no learning, no evaluation function, no Connect-Four knowledge
of any kind. And the dose-response curve: 800 dreams beat 50 dreams,
20–0. Imagination converts compute into strength, smoothly, on demand.

By the way — you have now met the Module 0 boss. That machine was
exactly this file's MCTS with a 3,000-dream budget. *No learning at
all.* You lost, in Module 0, to nothing but imagination. (Don't settle
the score yet. Your rematch should be against the machine that learns.)

## The crack in the wall

Watch a 400-dream MCTS think and the waste is physical: it spends
precious dreams on moves any club player rejects at a glance, because
**every move looks identical until imagined** — it has no instinct for
where to point the telescope. And its rollouts judge positions by
random-playout averages, which misjudge any position whose truth is
tactical, not statistical.

Inventory time. Module 7 built a network that *instantly* ranks moves
by smell — intuition, no foresight. Module 9 built search that
*carefully* verifies — foresight, no instinct. Module 8 built the
infinite teacher — self-play, which trains the one but could train on
the conclusions of the other.

Each piece patches the others' exact weakness. There is one machine left
to build in this course, and you already own all of its parts.

→ [`modules/10-reason`](../10-reason/README.md)
