# Module 7 — The Brain

> *Wall broken this module: THE wall — "memory can't cover the world."*

## One substitution

Take your Module 4 Q-learner. Find the table. Replace it with a neural
network. **Change nothing else.** The update target is still, character
for character, the bucket-brigade rule:

```
target = r + gamma * max_a' Q(s', a')
```

The only difference is what happens to the target once you have it. The
table *stored* it in a private universe. The network gets *nudged toward
it* — and because the network computes its answer from shared weights,
nudging it about one situation moves its opinion about **every situation
that looks similar**. The same gradient step that teaches (3,4)→(9,9)
leaks wisdom into (3,5)→(9,9), (4,4)→(9,9), and ten thousand situations
the agent will never visit. Generalization isn't a feature we add. It's
what stops being preventable once answers flow through shared knobs.

## Two stabilizers (learn them as scars, not trivia)

Strapping a network into the bucket brigade adds two real problems the
table never had, and your implementation carries one fix for each:

**Replay.** Consecutive steps of an episode are near-identical, and a
network trained on a stream of near-identical examples obsesses over
the present and forgets the past (the table couldn't forget — private
universes again). So: store experiences in a big buffer, train on
**random batches of old memories**. Dreams, basically — replaying
shuffled fragments of your past to consolidate them.

**Don't blame the target.** The target contains the network's own
output for `s'`. If blame flows into it, the network learns to make
targets easy instead of predictions right — the student grading its own
exam. Hence `.data` (a plain number, no graph) when computing targets.
One missing `.data` and training quietly chases its own tail; this bug
has eaten whole research weeks. Now it can't eat yours.

## Build

Open [`brain.py`](brain.py). Five TODOs: ask the net (`q_values`),
ε-greedy (`choose`), the memory buffer (`remember`), the batched bucket
brigade (`train_step`), and the life loop (`run_episode`).

```bash
pytest tests/test_module07.py -q       # ~1 minute; generalization is earned
python modules/07-the-brain/brain.py   # the main event
```

## What you should see

The demo returns to the scene of the crime: the same 16×16 moving-goal
world, the same honest exam on **fresh** situations. The Module 5
autopsy read: *table, 4,000 episodes, ~18%.* Now watch the column climb:

```
after  300 episodes: success on fresh exams  92%
after  600 episodes: success on fresh exams  99%
after 1200 episodes: success on fresh exams 100%
```

A perfect score, on situations it has never been in, with a third of the
table's experience. And here is the detail to savor: this network has
4,800 weights — it **couldn't** memorize 65,280 situations even if
memorizing were its plan. Being too small to remember the world is
precisely what forces it to do something better: notice that "walk
toward the goal" is one idea, not 65,280 facts. The constraint isn't a
compromise. The constraint is where the intelligence comes from.

The wall from Module 5 isn't dented. It's irrelevant.

## The reveal

You built **deep Q-learning** — a DQN, the algorithm with which
DeepMind learned Atari games from pixels in 2013 and lit the fuse of the
modern era; experience replay and target-decoupling are the two
stabilizers straight from that work. More importantly, you built it *as
a bug-fix*, the same way the field found it: not "neural networks are
powerful," but "tables die in big worlds and weight-sharing is the
rescue."

## The crack in the wall

Act II complete: your agent learns, generalizes, scales. Now notice the
quiet assumption every module so far has stood on — **the world sits
still.** Mazes don't fight back. Goals don't dodge.

Add an opponent and reward stops being a property of where you stand; it
depends on what *they* do next. And the deeper problem isn't strategy —
it's the curriculum. Against a fixed teacher you plateau at the
teacher's ceiling: a flawless mimic of mediocrity. Against whom does a
learner train to become *better than everyone available, including its
teachers*?

There's one opponent that is always exactly as good as you, improves at
exactly your pace, and never stops showing up.

→ [`modules/08-the-mirror`](../08-the-mirror/README.md)
