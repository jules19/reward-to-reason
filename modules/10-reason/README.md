# Module 10 — Reason

> *Wall broken this module: the last one — "intuition and imagination
> are separate faculties."*

## The wedding

Lay the parts on the bench:

- **Module 7**: a network that glances at a position and *feels* which
  moves are promising and who's winning. Fast, shallow, fallible.
- **Module 9**: a search that *checks*, by imagining futures. Careful,
  honest, blind — wastes its budget dreaming about garbage moves.
- **Module 8**: self-play — the teacher that is always exactly your
  level and never cancels.

Now the two design moves that fuse them into one machine. Both are
small. Together they made history.

**Move 1: put the brain inside the dream.** Take your MCTS and make two
deletions. Where it explored uniformly, let the network's *priors* steer
curiosity — promising moves get imagined first (the UCB rule grows a
prior and becomes PUCT). Where it rolled out randomly to the end of the
game, just *ask the value head* — no rollout at all. Intuition decides
where the telescope points; imagination does the verifying. This is
`GuidedMCTS`, and it is your Module 9 code with the casino lesson
upgraded and the daydreaming replaced by judgement.

**Move 2: close the loop.** Here is the engine of the whole thing, the
part to read twice:

> The search, *guided by the network*, produces better opinions than
> the raw network — it checked. So use the search's output as the
> **training target** for the network. Where did the search spend its
> visits? That's the policy target. How did the self-played game
> actually end? That's the value target.

The network teaches the search where to look; the search hands back
corrected judgement; the network absorbs it and guides better tomorrow.
**Intuition sharpens imagination; imagination corrects intuition.**
Around and around, with no teacher anywhere in the building — Module 8's
mirror supplying an endless stream of games at exactly the right
difficulty, forever.

Random static in, reason out. That's the claim. You're about to watch
it happen on your own hardware.

## Build

Open [`reason.py`](reason.py). Five TODOs, in dependency order:
`predict` and `train_batch` (the two-headed network), `_simulate` and
`_select_puct` (the brain goes into the dream), `self_play_game` (the
loop closes). The docstrings carry exact recipes; the thinking is in
understanding *why* each piece is what it is, and you've spent nine
modules earning that.

```bash
pytest tests/test_module10.py -q
python modules/10-reason/reason.py
```

## What you should see

A minute of self-play on tic-tac-toe — the network starting as pure
noise — examined against a random player and against *perfect play*:

```
iteration 10:  vs random 89/3/8    vs perfect-play 0/13/37
iteration 40:  vs random 89/0/11   vs perfect-play 0/9/41

Final exam: perfect player, 100 games, deep thought:
  0 wins / 0 losses / 100 draws
```

**A perfect player can no longer beat the thing that started as random
static an espresso ago.** Nobody showed it a single competent move. It
generated its own curriculum by arguing with itself.

## The boss fight

Tic-tac-toe is the rehearsal. Now:

```bash
python modules/10-reason/train_connect4.py            # start the campfire
python modules/10-reason/train_connect4.py --ladder   # the Hall of Mirrors
python modules/10-reason/train_connect4.py --play     # face your creation
```

The trainer checkpoints every few iterations and prints sparring results
against blind MCTS as it climbs. The `--ladder` command runs an Elo
tournament between **your engine's past selves** — watch yesterday's
checkpoint, which once impressed you, get calmly dismantled by today's.
Train as long as you care to; it resumes where it left off.

Then the rematch you've owed yourself since Module 0:

```bash
python -m r2r.play boss
```

The boss is blind MCTS with 3,000 dreams. Your engine dreams 10–40×
less and *sees*. When your trained engine beats the boss — and, let's
be honest about what's coming, when it beats *you* — take a second to
notice what you're feeling about a thing whose every line you typed.

## The reveal

This is **AlphaZero** — the loop, exactly, at desk-lamp scale: PUCT
search guided by a policy-value network, trained from its own search
visits and self-play outcomes. DeepMind's version (2017) used residual
towers, 5,000 TPUs, and 44 million games of self-play chess; within 24
hours it surpassed every chess engine humanity had built in fifty years
— and the *idea count* of what it did is the idea count of the file you
just wrote. The gap between your engine and that one is engineering —
silicon, parallelism, network depth. The gap between *no* engine and
your engine is the ten ideas you built one wall at a time. You now know
which gap is the deep one.

## The crack in the wall

There is no eleventh wall. Have you noticed what's left in your hands?
A primitive — reward — and a tower: preference, doubt, place, foresight,
blame, generalization, self-teaching, imagination, reason. Each layer
forced into existence by a measured failure of the one below. No layer
magic. The whole tower auditable, by you, because every floor of it is
*yours*.

One module remains. It contains no code. It's about where you — and the
machine you're talking to when you talk to one — actually are.

→ [`modules/11-you-are-here`](../11-you-are-here/README.md)
