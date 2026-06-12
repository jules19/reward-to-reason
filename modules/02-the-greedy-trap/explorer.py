"""Module 2 — The Greedy Trap.

Build two gamblers and discover why certainty is a losing strategy.

Read README.md first.
Check your work:   pytest tests/test_module02.py
See it live:       python modules/02-the-greedy-trap/explorer.py
"""

import random


class GreedyAgent:
    """Always exploits its current beliefs. Never doubts."""

    def __init__(self, n_actions, learning_rate=0.1, seed=None):
        # TODO 1: same memory as Module 1's coin —
        #   self.estimates, self.learning_rate, self._rng
        raise NotImplementedError("TODO 1")

    def choose(self):
        """Return the action with the highest estimate.
        Break ties RANDOMLY (with self._rng.choice) — otherwise the agent
        always starts with action 0 and your experiments will be biased."""
        # TODO 2
        raise NotImplementedError("TODO 2")

    def learn(self, action, reward):
        """Module 1's nudge rule, unchanged."""
        # TODO 3
        raise NotImplementedError("TODO 3")


class EpsilonGreedyAgent(GreedyAgent):
    """Mostly exploits — but with probability epsilon, tries something else.

    A single line of doubt is the difference between superstition and
    science.
    """

    def __init__(self, n_actions, learning_rate=0.1, epsilon=0.1, seed=None):
        super().__init__(n_actions, learning_rate, seed)
        self.epsilon = epsilon

    def choose(self):
        # TODO 4: with probability epsilon, return a completely random
        # action; otherwise do whatever GreedyAgent.choose does.
        # (self._rng.random() < self.epsilon is the coin flip you need.)
        raise NotImplementedError("TODO 4")


def lifetime_reward(env, agent, n_pulls=1000):
    """Run one agent's whole life in the casino. Returns total reward."""
    total = 0.0
    for _ in range(n_pulls):
        action = agent.choose()
        reward = env.pull(action)
        agent.learn(action, reward)
        total += reward
    return total


if __name__ == "__main__":
    from r2r.envs import Casino

    n_runs, n_pulls = 200, 1000
    greedy_total = explorer_total = 0.0
    greedy_found = explorer_found = 0
    for seed in range(n_runs):
        env = Casino(seed=seed)
        greedy = GreedyAgent(env.n_actions, seed=seed)
        explorer = EpsilonGreedyAgent(env.n_actions, epsilon=0.1, seed=seed)
        greedy_total += lifetime_reward(env, greedy, n_pulls)
        env = Casino(seed=seed)  # fresh but identical casino for fairness
        explorer_total += lifetime_reward(env, explorer, n_pulls)
        if max(range(env.n_actions), key=lambda i: greedy.estimates[i]) == env.best_action:
            greedy_found += 1
        if max(range(env.n_actions), key=lambda i: explorer.estimates[i]) == env.best_action:
            explorer_found += 1

    print(f"{n_runs} lifetimes in the casino, {n_pulls} pulls each\n")
    print(f"  pure greed : avg lifetime reward {greedy_total / n_runs:7.1f}, "
          f"found the best machine in {100 * greedy_found / n_runs:.0f}% of lives")
    print(f"  10% doubt  : avg lifetime reward {explorer_total / n_runs:7.1f}, "
          f"found the best machine in {100 * explorer_found / n_runs:.0f}% of lives")
    print("\nCertainty is a trap. The agents that doubt, win.")
