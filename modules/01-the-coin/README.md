# Module 1 — The Coin That Learns

> *Wall broken this module: "a number can't prefer anything."*

## The smallest question

Forget chess. Forget mazes. Forget neural networks. Here is the smallest
question about learning that is still a real question:

> A machine faces two buttons. One pays better than the other. Nobody
> tells it which. **Can it come to know?**

If intelligence can exist at all, it has to be able to exist here, in a
world with two buttons and no clock. This is our NAND gate moment: if we
find the primitive here, we get to spend nine modules composing it into
a mind.

## The world

`r2r.envs.CoinFlip` is the entire universe for this module:

```python
env = CoinFlip()
reward = env.pull(action)    # action is 0 or 1; reward is 1.0 or 0.0
```

Action 1 pays off 80% of the time, action 0 only 20% — but your agent
will never read that sentence. It sees only the rewards.

## The agent

Open [`coin_agent.py`](coin_agent.py). You are building `LearningCoin`,
a learner whose entire mental life is **two numbers**:

```
estimates[0]   "how good do I think action 0 is?"
estimates[1]   "how good do I think action 1 is?"
```

Both start at 0.0 — perfect ignorance. After every reward, exactly one
line of arithmetic runs:

```
estimate  <-  estimate + learning_rate * (reward - estimate)
```

Read that rule until it's boring, because **it is the only piece of
learning in this entire course**. Everything from here to the chess-like
engine in Module 10 is this line wearing progressively better disguises.

In words: *move your guess a fraction of the way toward what actually
happened.* The quantity `(reward - estimate)` is the **surprise** — the
gap between expectation and reality. When reality matches expectation,
surprise is zero, and learning stops on its own. The agent doesn't learn
from rewards; it learns from being *wrong* about rewards.

## Build

Four TODOs in `coin_agent.py`, top to bottom. Then:

```bash
pytest tests/test_module01.py -q     # the examiner
python modules/01-the-coin/coin_agent.py    # the show
```

## What you should see

The two estimates start at zero, jitter around, and then — without any
line of code that says "find the better button" — they *separate*. One
drifts up toward 0.8, the other settles near 0.2. The coin ends up
knowing the hidden truth of its universe, and its knowledge is just two
floats.

Run it a few times. Watch the early estimates be badly wrong (a lucky
streak on the bad button at the start can fool it for dozens of pulls)
and then watch reality grind the error away.

## What just happened (the reveal)

You built what the literature calls an **action-value estimate** updated
by an **exponential moving average**. The surprise term
`(reward - estimate)` has a grander name too: it is the seed of the
**temporal-difference error**, which you will meet again in Module 4 —
and which, as far as anyone can tell, is what dopamine neurons in your
own midbrain compute. The thing you just typed is not a metaphor for how
brains value things. It is the same computation.

## The crack in the wall

Notice what your coin does *not* do: it never decides anything. It flails
randomly (`explore()`) and merely *watches* — a critic, not an actor.
The moment we let it act on its beliefs, something rots. Suppose it acts
greedily from the start: it tries action 0, gets a lucky +1, estimates
become `[0.1, 0.0]` — and now action 0 looks better, so it pulls action
0, forever. One lucky coincidence hardens into superstition, and *being
sure* prevents it from ever finding out it's wrong.

A learner that acts on its beliefs needs a way to doubt them. That is a
genuinely new problem — and the next module.

→ [`modules/02-the-greedy-trap`](../02-the-greedy-trap/README.md)
