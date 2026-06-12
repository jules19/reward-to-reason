"""Module 1 solution — The Coin That Learns."""

import random


class LearningCoin:
    """The smallest learner that can exist: two numbers and a nudge rule.

    estimates[a] is the coin's current guess at "how good is action a?".
    Every reward nudges the guess for the action taken a little toward
    what actually happened. That nudge is the entire secret of this course.
    """

    def __init__(self, n_actions=2, learning_rate=0.1, seed=None):
        self.estimates = [0.0] * n_actions
        self.learning_rate = learning_rate
        self._rng = random.Random(seed)

    def explore(self):
        """Pick a random action (the coin just flips itself for now)."""
        return self._rng.randrange(len(self.estimates))

    def learn(self, action, reward):
        """Nudge the estimate for `action` toward the reward we just saw."""
        self.estimates[action] += self.learning_rate * (reward - self.estimates[action])

    def best_guess(self):
        """Which action does the coin now believe is better?"""
        best = max(self.estimates)
        return self.estimates.index(best)


def run_experiment(env, agent, n_pulls=300):
    """Let the agent flail randomly while learning. Returns the estimates."""
    for _ in range(n_pulls):
        action = agent.explore()
        reward = env.pull(action)
        agent.learn(action, reward)
    return agent.estimates


if __name__ == "__main__":
    from r2r.envs import CoinFlip

    env = CoinFlip(seed=42)
    agent = LearningCoin(seed=7)
    print("A coin with two numbers inside. Watch the numbers.\n")
    for step in range(1, 301):
        action = agent.explore()
        reward = env.pull(action)
        agent.learn(action, reward)
        if step in (1, 3, 10, 30, 100, 300):
            e = agent.estimates
            print(f"after {step:3d} flips: belief[heads]={e[0]:.2f}  belief[tails]={e[1]:.2f}")
    print(f"\nthe coin's verdict: action {agent.best_guess()} is better")
    print(f"the hidden truth:   action {env.best_action} is better")
    print("\nIt was never told. It learned. From nothing but reward.")
