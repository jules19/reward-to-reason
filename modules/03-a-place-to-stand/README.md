# Module 3 — A Place to Stand

> *Wall broken this module: "an agent with no concept of 'here' can't
> inhabit a world."*

## The maze

The world acquires geography. `r2r.envs.Maze` is a little labyrinth:

```
#########
#S..#...#
#.#.#.#.#
#.#...#.#
#.#####.#
#.....#.#
###.#.#.#
#...#..G#
#########
```

The agent starts at `S`. Four actions: up, right, down, left. Reaching
`G` ends the episode with reward **1.0**. Every other step of its life:
**0.0**. It cannot see the map above — it perceives exactly one thing,
its own coordinates, and it learns walls only by walking into them.

Stop and appreciate how brutal this is. Imagine learning a building
blindfolded, where the only feedback you ever receive is a single bell
that rings if you happen to stumble into one particular room. No
"warmer/colder". No partial credit. Reward is *silent* until the end.

## The one upgrade

Your casino agent kept `estimates[action]`. The new agent keeps:

```
q[(state, action)]    "how good is this action, FROM THIS PLACE?"
```

That dictionary is the **experience map**. Same nudge rule, same
ε-greedy doubt — just indexed by *where you are standing*. (Its official
name is a Q-table; you've now built one, so you're allowed to know
that.)

## The credit problem — solve it the obvious way

One new puzzle genuinely needs solving. A successful episode might be
80 steps long. The bell rings once, at the end. *Which of the 80 steps
deserves the credit?*

This module takes the honest, storyteller's answer: **all of them, with
the credit fading as you move back from the moment of success.** When an
episode ends in reward, walk back through the trajectory; the final step
gets nudged toward the full reward, the one before it toward
`discount ×` that, and so on, each step's target shrinking by the
discount factor (0.95 here). Steps near the triumph soak up strong
credit; the aimless wandering near the start gets almost none — which is
fair, because most of it didn't help.

And when an episode ends in *failure* (time runs out), the same arithmetic
quietly does something important — make sure you see it in your code:
the targets are all zero, so every step of a failed strategy gets
*decayed*. Optimism that doesn't pay rent gets evicted.

## Build

Open [`experience_map.py`](experience_map.py): `value`, `choose`,
`learn_from_episode`, `run_episode`.

```bash
pytest tests/test_module03.py -q
python modules/03-a-place-to-stand/experience_map.py
```

## What you should see

For the first dozens of episodes: nothing. Random staggering. The bell
rings occasionally by pure luck. Then the map starts to *condense* —
values crystallize backward from the goal like frost on a window — and
suddenly the printed greedy path is a clean 16-step needle through the
labyrinth.

The demo prints the route at the end. Nobody drew that route. **The
route is what the dictionary looks like after surprise has finished
hammering on it.**

## The reveal

You built **every-visit Monte Carlo learning over a Q-table**:
"Monte Carlo" because it learns from complete sampled episodes, the way
a gambler learns odds by playing whole hands. It is simple, correct,
and *unkillably honest* — it only ever learns from things that really
happened, all the way to their real conclusions.

## The crack in the wall

But honesty has a price, and it's hiding in plain sight. Your learner
treats the *episode* as the unit of experience. It cannot learn anything
in the middle of a story — and worse, what it learns about a step is
forever entangled with *everything that happened afterward in that
particular story*, including its own later mistakes.

Walk near a cliff edge once, stumble (because you were exploring,
ε-style) and fall: the Monte Carlo verdict poisons every step that led
there — *including the perfectly good ones*. The storyteller can't tell
"that place is dangerous" from "I happened to trip there once."

There is a way to learn what a place is *actually* worth, independent of
your own clumsiness passing through it. It requires something that will
feel slightly illegal the first time you write it: learning from your
own guesses.

→ [`modules/04-messages-from-the-future`](../04-messages-from-the-future/README.md)
