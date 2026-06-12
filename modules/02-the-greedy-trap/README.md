# Module 2 — The Greedy Trap

> *Wall broken this module: "acting on beliefs hardens lucky accidents
> into superstition."*

## The casino

The world grows: `r2r.envs.Casino` has **ten** machines, each paying
noisy rewards around a hidden average. One machine is the best. A
lifetime is 1,000 pulls. The agent now *acts on what it believes* —
no more free random flailing.

This tiny world contains, fully formed, one of the deepest dilemmas in
all of decision-making:

> Every pull spent on a machine you *believe* is worse is a pull wasted —
> unless your belief is wrong, in which case it's the most valuable pull
> of your life. You cannot know which, until after.

That's not a toy problem. It is a doctor allocating patients between a
proven treatment and a promising one; it is you deciding whether to
order the dish you always order. The casino is the smallest world where
*living well* and *learning the truth* pull in opposite directions.

## Build

Open [`explorer.py`](explorer.py). Two agents:

1. **`GreedyAgent`** — Module 1's estimates, but `choose()` always picks
   the current best. Build it honestly; it is not a strawman. It is what
   "just do what works" actually means as code.
2. **`EpsilonGreedyAgent`** — identical, plus *one line of doubt*: with
   probability ε (typically 0.1), ignore all beliefs and try something
   random.

One detail matters more than it looks: **break ties randomly**. A greedy
agent that resolves ties toward action 0 isn't neutral — it has a secret
prejudice, and your experiments will quietly measure the prejudice
instead of the strategy.

```bash
pytest tests/test_module02.py -q
python modules/02-the-greedy-trap/explorer.py
```

## What you should see

Two hundred lifetimes in the casino. The greedy agent finds the truly
best machine in roughly **half** its lives — a coin flip, after a
thousand pulls of evidence! It latches onto the first machine that pays
a few lucky rewards and spends the rest of its life confirming its own
first impression. It doesn't fail loudly; it fails *comfortably*,
collecting decent rewards from a mediocre machine while the best one
sits untouched three slots away.

The agent with 10% doubt finds the truth in the great majority of its
lives — and earns more along the way, despite *knowingly* wasting one
pull in ten.

Sit with that: **the agent that deliberately acts against its own best
judgement 10% of the time ends up both richer and wiser.** Doubt is not
a tax on competence. It is how competence is purchased.

## The reveal

You have just built **ε-greedy exploration** and run headfirst into the
**exploration–exploitation trade-off** — the dilemma that, more than any
other, separates reinforcement learning from ordinary supervised
learning. A supervised learner is handed its data. *Your* learner's
beliefs determine its actions, its actions determine its data, and its data
determine its beliefs. That loop is vicious by default. ε is the
simplest known way to keep it honest — and the not-so-simple ways
(you'll build one inside a search tree in Module 9, called UCB) are all
refinements of the same confession: *I might be wrong.*

## The crack in the wall

Now look at what your agent still cannot represent. Suppose the casino
had a rule: *machine 3 pays jackpots, but only right after you've played
machine 7.* Your agent could never even notice. It has opinions about
**actions**, but no concept of **circumstance** — no "after machine 7",
no "when I'm in the corridor", no *here*. Its whole model of the world
is a list of ten numbers floating in a void.

Real worlds have places, and what's good *here* is poison *there*. The
agent needs a memory organized by situation. That single upgrade — from
"how good is action a?" to "how good is action a **when things look like
this**?" — is the next module, and it cracks open everything: mazes,
games, and eventually chessboards.

→ [`modules/03-a-place-to-stand`](../03-a-place-to-stand/README.md)
