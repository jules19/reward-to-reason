"""Module 10 — Reason.

Everything you have built, fused into one machine:

  Module 1-4   reward and value            -> the training signal
  Module 6-7   networks and generalization -> the intuition
  Module 8     self-play                   -> the infinite teacher
  Module 9     tree search                 -> the imagination

A network suggests which futures are worth imagining; the search
imagines them and returns a better opinion than the network started
with; the better opinion becomes training data for the network.
Intuition sharpens imagination; imagination corrects intuition.
Around and around, with no teacher anywhere.

This is the AlphaZero loop, at desk-lamp scale. Read README.md first.

Check your work:   pytest tests/test_module10.py
See it live:       python modules/10-reason/reason.py
Boss fight:        python modules/10-reason/train_connect4.py
"""

import math
import random

import numpy as np

from r2r.tensor import Tensor, MLP, Adam


class PolicyValueNet:
    """One network, two opinions about any position:

      priors : "which moves smell promising here?"  (the intuition)
      value  : "how good is this position for me?"   (the judgement)

    Both are from the perspective of the player to move.
    """

    def __init__(self, game, hidden=64, lr=1e-3, seed=0):
        self.game = game
        rng = np.random.default_rng(seed)
        # One trunk, n_actions+1 outputs: action logits, then a value logit.
        self.net = MLP([game.encoded_size, hidden, hidden, game.n_actions + 1], rng)
        self.optimizer = Adam(self.net.parameters(), lr=lr)

    def predict(self, state):
        """Returns (priors, value) for one state.

          priors : dict {legal_move: probability}, summing to 1 — a
                   softmax over the LEGAL moves' logits only
          value  : tanh of the last output, a number in [-1, 1]

        Recipe: out = self.net(np.array([game.encode(state)])).data[0];
        the first n_actions entries are move logits, out[-1] is the value
        logit. Subtract the max logit before exponentiating (numerical
        hygiene).
        """
        # TODO 1
        raise NotImplementedError("TODO 1")

    def train_batch(self, states, target_policies, target_values):
        """One gradient step on a batch. Loss = policy loss + value loss.

          value loss  : (tanh(value_logit) - z)^2, summed over the batch
          policy loss : cross-entropy, -sum(target * log_softmax(logits))

        The network's output mixes action logits and the value logit in
        one row, and a Tensor has no slicing — use masks:

          policy_mask : 1.0 on the n_actions logit columns, 0 on the value
          value_mask  : the reverse

          value head  : value_out = (out * Tensor(value_mask)).tanh()
                        compare against targets placed in the value column
          policy head : push the value column's logit out of the softmax by
                        adding (policy_mask - 1) * 1e9 to the masked logits,
                        then .log_softmax(), then weight by the targets.

        Finish with zero_grad / backward / step. Return the average loss
        per example (a plain float) for logging.
        """
        # TODO 2
        raise NotImplementedError("TODO 2")


class GuidedMCTS:
    """Module 9's tree search with Module 7's brain inside.

    Two changes from your MCTS, and only two:
      1. No rollouts. At a leaf, ask the network's value head instead of
         daydreaming randomly to the end.
      2. Curiosity is steered by the network's priors (the PUCT rule):
         moves that smell promising get imagined first.
    """

    def __init__(self, net, n_simulations=50, c_puct=1.5, root_noise=0.25,
                 seed=None):
        self.net = net
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.root_noise = root_noise
        self._rng = random.Random(seed)

    def search(self, game, state):
        """Run simulations, return {move: visit_count} for the root."""
        visits, totals, priors = {}, {}, {}
        root_priors, _ = self.net.predict(state)
        if self.root_noise > 0:
            # Blend in uniform noise at the root so self-play keeps
            # discovering moves the current intuition underrates.
            k = len(root_priors)
            root_priors = {m: (1 - self.root_noise) * p + self.root_noise / k
                           for m, p in root_priors.items()}
        priors[state] = root_priors
        for _ in range(self.n_simulations):
            self._simulate(game, state, visits, totals, priors)
        return {m: visits.get((state, m), 0) for m in game.legal_moves(state)}

    def select_move(self, game, state):
        counts = self.search(game, state)
        best = max(counts.values())
        return self._rng.choice([m for m, c in counts.items() if c == best])

    def _simulate(self, game, state, visits, totals, priors):
        """One guided imagining. Like Module 9's _simulate, except:

        Descend by _select_puct while the state is in `priors` (i.e. the
        tree). Stop when you hit either:
          * a terminal state  -> value_abs = the actual winner (float)
          * an unseen state   -> ask the net: p, v = self.net.predict(state)
                                 store priors[state] = p
                                 value_abs = v * current_player(state)
                                 (v is from the mover's perspective;
                                  multiplying converts it to +1's view)
        NO rollout phase. Then backpropagate value_abs along the path
        exactly as in Module 9.
        """
        # TODO 3
        raise NotImplementedError("TODO 3")

    def _select_puct(self, game, state, visits, totals, priors):
        """The PUCT rule. For each legal move m:

            q = totals / visits     (0 if unvisited)
            u = c_puct * priors[state][m] * sqrt(N_total + 1) / (1 + visits)

        where N_total is the sum of this state's move visits. Return the
        move maximizing q + u. Note how curiosity (u) is now WEIGHTED BY
        INTUITION — the brain tells the dreamer where to look.
        """
        # TODO 4
        raise NotImplementedError("TODO 4")


class AlphaLite:
    """The full loop: self-play -> train -> stronger self-play -> ..."""

    def __init__(self, game, hidden=64, n_simulations=50, lr=1e-3,
                 temperature_moves=4, buffer_size=20000, batch_size=64,
                 seed=0):
        self.game = game
        self.net = PolicyValueNet(game, hidden=hidden, lr=lr, seed=seed)
        self.n_simulations = n_simulations
        self.temperature_moves = temperature_moves
        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self._rng = random.Random(seed)

    def self_play_game(self):
        """Play one game against the mirror, with imagination on.

        Create ONE GuidedMCTS for the game (seed it with self._rng.random()).
        At every position until the game ends:

          1. counts = mcts.search(game, state)
          2. record a training example:
               ( game.encode(state),
                 policy: np.zeros(n_actions) with counts normalized into it,
                 game.current_player(state) )
          3. pick the move:
               * first `temperature_moves` moves: SAMPLE proportionally to
                 counts (self._rng.choices) — varied openings, so the
                 mirror never stops surprising itself
               * after that: the most-visited move
          4. step.

        When the game ends with winner z, convert each recorded example
        into (encoded, policy, z * player) — "how did this game end, from
        the perspective of whoever was to move?" — and append them all to
        self.buffer (dropping oldest beyond buffer_size).
        """
        # TODO 5
        raise NotImplementedError("TODO 5")

    def train_iteration(self, n_games=10, n_batches=20):
        """One turn of the crank: self-play, then gradient steps on
        random batches from the buffer."""
        for _ in range(n_games):
            self.self_play_game()
        losses = []
        for _ in range(n_batches):
            if len(self.buffer) < self.batch_size:
                break
            batch = self._rng.sample(self.buffer, self.batch_size)
            losses.append(self.net.train_batch(
                [b[0] for b in batch],
                np.array([b[1] for b in batch]),
                np.array([b[2] for b in batch]),
            ))
        return sum(losses) / len(losses) if losses else float("nan")

    def player(self, n_simulations=None, seed=0):
        """A frozen agent using the current net (for the arena)."""
        return GuidedMCTS(self.net, n_simulations or self.n_simulations,
                          root_noise=0.0, seed=seed)


if __name__ == "__main__":
    import time

    from r2r.arena import play_match
    from r2r.builtin import MinimaxAgent, RandomAgent
    from r2r.envs import TicTacToe

    game = TicTacToe()
    engine = AlphaLite(game, hidden=64, n_simulations=50,
                       temperature_moves=4, buffer_size=3000, seed=0)

    print("The loop: imagine -> play yourself -> learn -> imagine better.")
    print("(About a minute of training. Watch the exam columns.)\n")
    t0 = time.time()
    for iteration in range(1, 41):
        engine.train_iteration(n_games=10, n_batches=30)
        if iteration % 10 == 0:
            agent = engine.player(n_simulations=100)
            w, l, d = play_match(game, agent, RandomAgent(seed=1), n_games=100)
            wm, lm, dm = play_match(game, agent, MinimaxAgent(randomize=7),
                                    n_games=50)
            print(f"  iteration {iteration:2d} ({time.time() - t0:3.0f}s): "
                  f"vs random {w}/{l}/{d} | vs perfect-play {wm}/{lm}/{dm}")

    print("\nFinal exam: a PERFECT player, 100 games, and the engine gets")
    print("more thinking time (800 imagined futures per move)...")
    agent = engine.player(n_simulations=800)
    wm, lm, dm = play_match(game, agent, MinimaxAgent(randomize=7), n_games=100)
    print(f"  result: {wm} wins / {lm} losses / {dm} draws")

    print("\nThe network in this engine started as random static. Nobody fed")
    print("it a single example of good play. It generated its own curriculum")
    print("by arguing with itself — search correcting intuition, intuition")
    print("focusing search — until a perfect player could no longer beat it.")
    print("\nNow run the same loop on Connect Four:")
    print("  python modules/10-reason/train_connect4.py")
    print("then play your own creation, and finally settle the score:")
    print("  python -m r2r.play boss")
