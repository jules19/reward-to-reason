# Module 8 — The Mirror

> *Wall broken this module: "a learner can't outgrow its teacher."*

## The room with no teacher

Lock the agent in a room with a tic-tac-toe board and nothing else. No
opponent database. No strategy hints. No teacher. It plays **both
sides** of every game, and when a game ends, the world emits its entire
opinion: `+1`, `-1`, or `0`. From this — only this — it must become an
expert.

It sounds like cheating, or a Zen koan: where could the knowledge
possibly *come from*? Hold that question; the demo answers it.

## The machinery (you own all of it already)

One value table, `values[state]` — "how will this game end, from player
+1's point of view?" — updated by, of course, the nudge rule:

```
values[s] <- values[s] + lr * (final_result - values[s])
```

Module 3's storyteller, transplanted to a board. (Why the storyteller,
and not Module 4's bootstrapping? Games are short, they always end, and
the final result is *exactly* the thing we want to predict — when whole
true stories are cheap, learn from whole true stories. Choosing tools by
the shape of the problem is graduation-level judgement; notice yourself
doing it.)

One genuinely new wrinkle — the only one: **the mirror plays both
colors.** Values are stored from +1's perspective, but on half the
moves, the mover is -1, who wants the *most negative* future. One
careful sign flip (`player * value`) makes the same table serve both
sides. The tests will catch you if you fumble it; nearly everyone
fumbles it once.

## Build

Open [`mirror.py`](mirror.py): `value` (terminal positions are facts,
not opinions), `best_move` (the sign flip), `play_training_game`
(ε-greedy against yourself), `learn_from_game` (the nudge).

```bash
pytest tests/test_module08.py -q
python modules/08-the-mirror/mirror.py
```

## What you should see

Thirty thousand mirror games, about a minute. The agent is examined by
two outsiders it has never met: a random player, and a **perfect**
minimax player (`r2r.builtin` — yes, you build minimax yourself next
module; today it's the examiner).

```
after  5000 mirror games:  vs random 188/0/12    vs PERFECT play 0/0/2
after 30000 mirror games:  vs random 184/0/16    vs PERFECT play 0/0/2
```

Read the right column again. **Zero losses against perfect play.**
Tic-tac-toe's saving grace is that draws are always available to
flawless defense — and the mirror found flawless defense *without ever
meeting an opponent*.

So: where did the knowledge come from? Watch the loop closely and the
koan dissolves. The board's *rules* silently label every terminal
position — `value()` treats wins as facts, remember. Self-play is a pump
that moves those facts backward into the opening: yesterday's self
blunders, today's self punishes the blunder, the result stamps every
position on the path, and tomorrow's self inherits positions already
marked "this ends badly." The agent isn't learning *from* the mirror.
It's using the mirror to **interrogate the rules** — reward bouncing
between two copies of itself until the game's own structure has been
squeezed into the table. The knowledge was in the rules all along.
Self-play is how you get it out.

And the curriculum is automatic and *perfect*: the opponent is never too
weak to punish you, never too strong to learn from, exactly at your
level for every minute of your career — Vygotsky's zone of proximal
development, implemented in four lines, no Vygotsky required.

## The reveal

This is **self-play reinforcement learning** — the engine of TD-Gammon
(1992), which shocked backgammon grandmasters from a mirror and a value
function, and of AlphaGo Zero (2017), which did the same to Go. It is
also, give or take, the *first program in Sutton & Barto's textbook* —
the tic-tac-toe learner of chapter one. You've now built the field's
alpha and omega, and they turn out to be the same program.

## The crack in the wall

Now scale the board. Connect Four: ~10¹³ positions. Your mirror learner
stores *opinions about positions it has visited* — a table. You
executed tables in Module 5. You know how this ends.

"Use the Module 7 network instead of the table" — right instinct,
half the answer. A network whispering instant judgements is what a
strong player calls *intuition*, and intuition alone plays fast,
shallow, occasionally catastrophic chess. The other half of strength is
what the grandmaster does next: sits on her hands and **checks** —
"if I go there, he goes there..." — spending thought *before* the move,
at decision time, not training time.

We have no machinery for that. Next module builds thinking-ahead from
scratch — two kinds — and the module after that performs the wedding.

→ [`modules/09-imagination`](../09-imagination/README.md)
