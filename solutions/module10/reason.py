"""Module 10 solution — Reason.

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

This is the AlphaZero loop, at desk-lamp scale.
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
        """Returns (priors dict over legal moves, value in [-1, 1])."""
        out = self.net(np.array([self.game.encode(state)])).data[0]
        legal = self.game.legal_moves(state)
        logits = np.array([out[m] for m in legal])
        logits -= logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()
        priors = {m: float(p) for m, p in zip(legal, probs)}
        value = float(np.tanh(out[-1]))
        return priors, value

    def train_batch(self, states, target_policies, target_values):
        """One gradient step. Loss = cross-entropy(policy) + MSE(value)."""
        n = len(states)
        xs = np.array(states)
        out = self.net(xs)

        # Split the combined head with masks (a Tensor has no slicing —
        # masks keep the autograd graph simple).
        n_act = self.game.n_actions
        policy_mask = np.zeros((n, n_act + 1))
        policy_mask[:, :n_act] = 1.0
        value_mask = np.zeros((n, n_act + 1))
        value_mask[:, n_act] = 1.0

        # Value loss: (tanh(value_logit) - z)^2
        value_out = (out * Tensor(value_mask)).tanh()
        value_target = np.zeros((n, n_act + 1))
        value_target[:, n_act] = target_values
        value_diff = value_out * Tensor(value_mask) - Tensor(value_target)
        value_loss = (value_diff * value_diff).sum()

        # Policy loss: -sum(target * log_softmax(logits)). The mask trick:
        # push non-action logits far down so they can't steal probability.
        masked_logits = out * Tensor(policy_mask) + Tensor((policy_mask - 1.0) * 1e9)
        logp = masked_logits.log_softmax()
        targets = np.zeros((n, n_act + 1))
        targets[:, :n_act] = target_policies
        policy_loss = -(logp * Tensor(targets)).sum()

        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.data) / n


class GuidedMCTS:
    """Module 9's tree search with Module 7's brain inside.

    Two changes from your MCTS, and only two:
      1. No rollouts. At a leaf, ask the network's value head instead of
         daydreaming randomly to the end.
      2. Curiosity is steered by the network's priors (PUCT): moves that
         smell promising get imagined first.
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
            self._simulate(game, state, visits, totals, priors, root=state)
        return {m: visits.get((state, m), 0) for m in game.legal_moves(state)}

    def select_move(self, game, state):
        counts = self.search(game, state)
        best = max(counts.values())
        return self._rng.choice([m for m, c in counts.items() if c == best])

    def _simulate(self, game, state, visits, totals, priors, root=None):
        path = []
        while True:
            winner = game.winner(state)
            if winner is not None:
                value_abs = float(winner)  # terminal: the truth, no guessing
                break
            if state not in priors:
                # Leaf: ask the brain, don't dream.
                p, value = self.net.predict(state)
                priors[state] = p
                value_abs = value * game.current_player(state)
                break
            move = self._select_puct(game, state, visits, totals, priors)
            path.append((state, move))
            state = game.next_state(state, move)
        for s, m in path:
            mover = game.current_player(s)
            visits[(s, m)] = visits.get((s, m), 0) + 1
            totals[(s, m)] = totals.get((s, m), 0.0) + value_abs * mover

    def _select_puct(self, game, state, visits, totals, priors):
        moves = game.legal_moves(state)
        sqrt_n = math.sqrt(sum(visits.get((state, m), 0) for m in moves) + 1)

        def puct(m):
            n = visits.get((state, m), 0)
            q = totals.get((state, m), 0.0) / n if n else 0.0
            u = self.c_puct * priors[state][m] * sqrt_n / (1 + n)
            return q + u

        return max(moves, key=puct)


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

        Every position contributes a training example:
          input  : the encoded position
          policy : where the SEARCH ended up spending its imagination
                   (a better opinion than the raw network's)
          value  : how the game actually ended, for the player to move
        """
        mcts = GuidedMCTS(self.net, self.n_simulations, seed=self._rng.random())
        state = self.game.initial_state()
        records = []
        move_number = 0
        while self.game.winner(state) is None:
            counts = mcts.search(self.game, state)
            total = sum(counts.values())
            policy = np.zeros(self.game.n_actions)
            for m, c in counts.items():
                policy[m] = c / total
            records.append((self.game.encode(state), policy,
                            self.game.current_player(state)))
            if move_number < self.temperature_moves:
                # Early moves: sample, so the mirror sees varied games.
                moves, weights = zip(*counts.items())
                move = self._rng.choices(moves, weights=[c + 1e-9 for c in weights])[0]
            else:
                move = max(counts, key=counts.get)
            state = self.game.next_state(state, move)
            move_number += 1
        z = float(self.game.winner(state))
        for encoded, policy, player in records:
            self.buffer.append((encoded, policy, z * player))
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)

    def train_iteration(self, n_games=10, n_batches=20):
        """One turn of the crank."""
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
