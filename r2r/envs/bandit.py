"""Bandit environments — the smallest worlds in which learning can exist.

A bandit has no states and no time. There is only: act, receive reward.
If learning can happen anywhere, it can happen here.
"""

import random


class CoinFlip:
    """Module 1's world. Two actions. One of them is secretly better.

    Action 0 pays +1 with probability 0.2.
    Action 1 pays +1 with probability 0.8.

    The agent is never told this. It only sees rewards.
    """

    n_actions = 2

    def __init__(self, seed=None):
        self._rng = random.Random(seed)
        self._probs = [0.2, 0.8]
        self.best_action = 1

    def pull(self, action):
        """Take an action, receive a reward. The entire interface."""
        if action not in (0, 1):
            raise ValueError(f"action must be 0 or 1, got {action!r}")
        return 1.0 if self._rng.random() < self._probs[action] else 0.0


class Casino:
    """Module 2's world. Ten slot machines with hidden payout rates.

    Each arm i pays a reward drawn from a normal distribution centred on a
    hidden value mu_i. One arm is best. A purely greedy agent will usually
    latch onto the first arm that happens to pay well — and never discover
    the truth.
    """

    n_actions = 10

    def __init__(self, seed=None):
        self._rng = random.Random(seed)
        self._mus = [self._rng.gauss(0.0, 1.0) for _ in range(self.n_actions)]
        self.best_action = max(range(self.n_actions), key=lambda i: self._mus[i])
        self.best_mean = self._mus[self.best_action]

    def pull(self, action):
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions}), got {action!r}")
        return self._rng.gauss(self._mus[action], 1.0)
