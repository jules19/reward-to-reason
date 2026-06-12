# Module 4 — Messages From the Future

> *Wall broken this module: "you can only learn from finished stories."*

## The illegal move

Here is the idea, and it should bother you:

> When you step from state `s` to state `s'`, don't wait to find out how
> the story ends. Update your estimate of `s` **using your current
> estimate of `s'`**.

```
q[s, a]  <-  q[s, a] + lr * ( r + gamma * max_a' q[s', a'] - q[s, a] )
```

Module 1's nudge rule again — but look at the target. It contains
`max q[s', ·]`, which is *your own guess*. You are treating a belief as
if it were evidence. Learning a guess from a guess. Why isn't this a
circular fantasy that spirals off into self-confirming nonsense?

Because of one anchor: **at the goal, the value isn't a guess.** The
step that reaches the goal learns from the real reward — a fact. Then
the state one step earlier learns from *that* state's value, which is
now slightly factual. Then the state before that. Episode after episode,
truth crawls backward from the goal, one handshake at a time, until the
whole maze glows with directions. The guesses don't float free; they
form a bucket brigade with reality at one end.

(There is real math guaranteeing this converges — the Bellman equation;
see the reveal — but build it first and watch it work. The bucket
brigade *is* the proof, animated.)

## What this buys you

Two enormous things, one obvious and one subtle:

**You learn during the episode, from every single step** — including
episodes that end in failure, including episodes that never end at all.
The trajectory list is gone from your code. That's not a refactor;
that's the agent no longer needing stories.

**You learn what the world is, not what happened to you.** Look at the
target again: `max` over next actions. It asks "what's the best that
could follow from here?" — not "what did I happen to do next?" Your
exploring self might stumble after reaching `s'` (ε makes you trip one
step in ten!), but your *estimate* of `s'` is untouched by your own
clumsiness. Module 3's storyteller can't do this — its targets are
whole lived stories, stumbles included.

## Build, then watch the cliff

Open [`q_learner.py`](q_learner.py). Five TODOs — `learn()` is the
sacred one.

```bash
pytest tests/test_module04.py -q
python modules/04-messages-from-the-future/q_learner.py
```

The demo stages a duel on the **cliff walk**: a ledge where the fast
route to the goal hugs a deadly edge, and a long safe route loops around
the top. Both your Module 3 agent and this one train for 5,000 episodes,
same world, same reward, same dice. Then each walks its best path,
soberly, no exploration.

The storyteller takes the long way around — **it has learned to fear the
edge**, because its memories of the edge are contaminated with its own
exploratory stumbles into the void. Your new learner walks the brink,
13 steps, optimal — not because it's brave, but because it learned the
cliff's actual value rather than its own biography there.

Same information. Different question asked of it. Different character.

## The reveal

You have just invented, in order: **temporal-difference learning**
(learning guesses from guesses), **off-policy learning** (learning about
the best behaviour while practicing exploratory behaviour — the `max` did
that), and the algorithm combining them: **Q-learning**, published by
Chris Watkins in 1989 and still the conceptual spine of the field. The
self-consistency condition your bucket brigade settles into —
*a state's value equals immediate reward plus discounted value of the
best successor* — is the **Bellman equation**, the closest thing
decision theory has to a law of nature. You didn't memorize it. You
wrote it as a bug-fix for slow mazes. That's the right way around.

One more thing, and it isn't hype: that TD error your code computes,
`r + gamma·max q' − q`? Recordings of midbrain dopamine neurons show
them signalling almost exactly this quantity — firing on unexpected
reward, silent on expected reward, dipping on disappointment. The most
load-bearing line in your file is currently the best computational
theory of what dopamine *is*.

## The crack in the wall

Act I is complete: doubt, place-memory, foresight — a genuine learner.
Time to be cruel to it.

Every world so far had one property your agent never noticed it was
relying on: **places repeat**. Visit (3,4) enough times and the nudges
average into wisdom. But what if the world never holds still — what if
no situation, once lived, ever comes back exactly? A chessboard, for
instance. Or anything real.

The next module doesn't teach a technique. It builds the gallows,
measures the rope, and hangs the Q-table in public. Bring your own
instruments — you'll write the measuring tools yourself.

→ [`modules/05-the-wall`](../05-the-wall/README.md)
