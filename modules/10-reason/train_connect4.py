"""The boss run: AlphaZero-lite on Connect Four.

This is not a unit test. It is a campfire. Start it, make tea, and watch
an intelligence assemble itself out of a reward signal:

    python modules/10-reason/train_connect4.py

Checkpoints are saved to modules/10-reason/checkpoints/ every few
iterations. While it trains (or after), in another terminal:

    python modules/10-reason/train_connect4.py --ladder   # Elo: all your past selves
    python modules/10-reason/train_connect4.py --play     # face your creation

Budget: with the default settings an iteration takes ~10-30 seconds on
a plain laptop. After ~10 minutes (60 iterations) the engine already
beats blind MCTS at EQUAL thinking budget — intuition paying rent.
Catching the 3000-dream Module 0 boss takes real training and maybe
some tuning of your own. That fight is the course's final exercise,
and nobody is going to promise you the ending.
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from r2r.arena import Ladder, play_match
from r2r.builtin import MCTSAgent, RandomAgent
from r2r.course import load
from r2r.envs import ConnectFour

reason = load(10, "reason")

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

TRAIN = dict(hidden=128, n_simulations=80, temperature_moves=8,
             buffer_size=30000, batch_size=64, lr=1e-3, seed=0)
GAMES_PER_ITERATION = 16
BATCHES_PER_ITERATION = 60
CHECKPOINT_EVERY = 5


def save_checkpoint(engine, iteration):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    weights = [p.data.copy() for p in engine.net.net.parameters()]
    with open(CHECKPOINT_DIR / f"iter{iteration:04d}.pkl", "wb") as f:
        pickle.dump({"weights": weights, "config": TRAIN, "iteration": iteration}, f)


def load_engine(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    cfg = {k: v for k, v in data["config"].items()
           if k in ("hidden", "n_simulations", "temperature_moves",
                    "buffer_size", "batch_size", "lr", "seed")}
    engine = reason.AlphaLite(ConnectFour(), **cfg)
    for p, w in zip(engine.net.net.parameters(), data["weights"]):
        p.data = np.array(w)
    return engine


def checkpoints():
    return sorted(CHECKPOINT_DIR.glob("iter*.pkl"))


def train(iterations):
    game = ConnectFour()
    paths = checkpoints()
    if paths:
        engine = load_engine(paths[-1])
        start = int(paths[-1].stem[4:]) + 1
        print(f"resuming from {paths[-1].name}")
    else:
        engine = reason.AlphaLite(game, **TRAIN)
        start = 1
    t0 = time.time()
    for it in range(start, start + iterations):
        loss = engine.train_iteration(n_games=GAMES_PER_ITERATION,
                                      n_batches=BATCHES_PER_ITERATION)
        line = (f"iteration {it:3d} | loss {loss:.3f} | "
                f"buffer {len(engine.buffer):,} | {time.time() - t0:5.0f}s")
        if it % CHECKPOINT_EVERY == 0 or it == start + iterations - 1:
            save_checkpoint(engine, it)
            w, l, d = play_match(game, engine.player(n_simulations=80, seed=it),
                                 MCTSAgent(n_simulations=80, seed=it), n_games=20)
            line += f" | vs blind-MCTS(80): {w}/{l}/{d}  [checkpoint saved]"
        print(line, flush=True)
    print("\nDone. Now:")
    print("  python modules/10-reason/train_connect4.py --ladder")
    print("  python modules/10-reason/train_connect4.py --play")


def ladder():
    """The Hall of Mirrors: every past self, ranked."""
    paths = checkpoints()
    if not paths:
        print("no checkpoints yet — train first")
        return
    game = ConnectFour()
    lad = Ladder(game)
    lad.add("random", RandomAgent(seed=0))
    lad.add("blind-mcts-80", MCTSAgent(n_simulations=80, seed=0))
    if len(paths) <= 5:
        keep = paths
    else:  # first, last, and three evenly spaced between
        idx = sorted({round(i * (len(paths) - 1) / 4) for i in range(5)})
        keep = [paths[i] for i in idx]
    for path in keep:
        engine = load_engine(path)
        lad.add(f"you@{path.stem}", engine.player(n_simulations=80, seed=1))
    print("Playing the Hall of Mirrors (a few minutes)...\n")
    lad.run(games_per_pair=20, verbose=True)
    print()
    print(lad.table())
    print("\nEvery 'you@' line is a former self that today's engine outgrew.")


def play():
    paths = checkpoints()
    if not paths:
        print("no checkpoints yet — train first")
        return
    engine = load_engine(paths[-1])
    agent = engine.player(n_simulations=300, seed=int(time.time()))
    game = ConnectFour()
    from r2r.play import HumanAgent
    human, state = HumanAgent(), game.initial_state()
    print(f"You vs your own creation ({paths[-1].name}). You are X.\n")
    print(game.render(state) + "\n")
    while game.winner(state) is None:
        if game.current_player(state) == 1:
            move = human.select_move(game, state)
        else:
            move = agent.select_move(game, state)
            print(f"your creation plays {move}")
        state = game.next_state(state, move)
        print(game.render(state) + "\n")
    result = {0: "Draw.", 1: "You beat your own creation. Train it more.",
              -1: "Your creation won. You built the thing that beat you."}
    print(result[game.winner(state)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--ladder", action="store_true")
    parser.add_argument("--play", action="store_true")
    args = parser.parse_args()
    if args.ladder:
        ladder()
    elif args.play:
        play()
    else:
        train(args.iterations)
