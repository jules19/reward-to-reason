"""Module 1 — The Coin That Learns.

Build the smallest learner that can exist: two numbers and a nudge rule.

Read README.md first. Then fill in the four TODOs below, top to bottom.
Check your work:   pytest tests/test_module01.py
See it live:       python modules/01-the-coin/coin_agent.py
"""

import random


class LearningCoin:
    """A learner made of two numbers.

    estimates[a] is the coin's current guess at "how good is action a?".
    It starts knowing nothing (both zero). Every reward nudges the guess
    for the action taken a little toward what actually happened.
    """

    def __init__(self, n_actions=2, learning_rate=0.1, seed=None):
        # TODO 1: give the coin its memory.
        #   self.estimates       -> a list of n_actions zeros
        #   self.learning_rate   -> store it
        #   self._rng            -> random.Random(seed)
        raise NotImplementedError("TODO 1: build the coin's memory")

    def explore(self):
        """Pick a random action. (The coin just flails, for now —
        ACTING on its beliefs is Module 2's problem.)"""
        # TODO 2: return a random action index using self._rng.randrange.
        raise NotImplementedError("TODO 2: flail")

    def learn(self, action, reward):
        """The nudge rule — the only line of true learning in this course.
        Everything later is this line wearing better and better disguises:

            estimate <- estimate + learning_rate * (reward - estimate)

        In words: move the guess a fraction of the way toward what just
        actually happened. The gap (reward - estimate) is called surprise.
        No surprise, no learning.
        """
        # TODO 3: apply the nudge rule to self.estimates[action].
        raise NotImplementedError("TODO 3: the nudge rule")

    def best_guess(self):
        """Which action does the coin now believe is better?"""
        # TODO 4: return the index of the largest estimate.
        raise NotImplementedError("TODO 4: the verdict")


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
