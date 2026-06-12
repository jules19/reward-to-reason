"""Module 7 — The Brain.

Replace the table with a function. Same update rule as Module 4; a
profoundly different kind of memory.

Read README.md first. You have now earned r2r.tensor (the Built-In Chip
Rule) — it is the Dial you built, vectorized.

Check your work:   pytest tests/test_module07.py
See it live:       python modules/07-the-brain/brain.py
"""

import random

import numpy as np

from r2r.tensor import Tensor, MLP, Adam


class DeepQ:
    """Q-learning where the table is replaced by a neural network.

    The update rule is EXACTLY Module 4's:

        target = r + gamma * max_a' Q(s', a')

    The only change is what we do with the target. A table would store
    it. We instead nudge the network's weights so its output moves toward
    it — and because nearby states share weights, every nudge teaches the
    network about thousands of states it has never seen.
    """

    def __init__(self, state_size, n_actions, hidden=64, lr=1e-3,
                 gamma=0.95, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay_steps=12000, buffer_size=20000,
                 batch_size=64, warmup=500, seed=0):
        rng_np = np.random.default_rng(seed)
        self.net = MLP([state_size, hidden, hidden, n_actions], rng_np)
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_step = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.warmup = warmup
        self._rng = random.Random(seed)

    def q_values(self, encoded_state):
        """Ask the network for all action values of one encoded state.

        self.net takes a 2D array (a BATCH of states) and returns a
        Tensor. For a single state, wrap it: np.array([encoded_state]),
        then read .data[0] of the result.
        """
        # TODO 1
        raise NotImplementedError("TODO 1")

    def choose(self, encoded_state):
        """Epsilon-greedy over q_values. (np.argmax breaks ties for you.)"""
        # TODO 2
        raise NotImplementedError("TODO 2")

    def remember(self, state, action, reward, next_state, terminal):
        """Append one experience tuple to self.buffer; drop the OLDEST
        entry if the buffer exceeds self.buffer_size.

        `terminal` means next_state has no future (the goal was reached —
        NOT that time merely ran out)."""
        # TODO 3
        raise NotImplementedError("TODO 3")

    def train_step(self):
        """One nudge of the weights, from one random batch of memories.

        Do nothing until the buffer holds max(warmup, batch_size) items.
        Then:
          1. batch = self._rng.sample(self.buffer, self.batch_size)
          2. Module 4's target, vectorized over the batch:
                 next_best = self.net(next_states).data.max(axis=1)
                 targets   = rewards + gamma * next_best * (1 - terminal)
             (use the .data — the target is a NUMBER to chase, blame
              must not flow through it)
          3. The loss: squared error between the network's output FOR THE
             ACTIONS ACTUALLY TAKEN and the targets. Build it with a mask
             so the autograd graph stays simple:
                 mask[i, actions[i]] = 1.0
                 diff = out * Tensor(mask) - Tensor(mask * targets[:, None])
                 loss = (diff * diff).sum()
          4. zero_grad, backward, step.
        """
        # TODO 4
        raise NotImplementedError("TODO 4")

    def run_episode(self, env):
        """One life: act, remember, train_step, decay epsilon — every step.

        Decay: self.epsilon = max(epsilon_end, epsilon - epsilon_step).
        Remember to encode states with env.encode(...) before storing or
        choosing. Return total reward."""
        # TODO 5
        raise NotImplementedError("TODO 5")

    def train(self, env, episodes, report_every=None):
        for ep in range(episodes):
            self.run_episode(env)
            if report_every and (ep + 1) % report_every == 0:
                yield ep + 1


if __name__ == "__main__":
    from r2r.course import load
    from r2r.envs import OpenWorld

    wall = load(5, "the_wall")

    env = OpenWorld(size=16, seed=7)
    agent = DeepQ(state_size=4, n_actions=env.n_actions, seed=0)

    print(f"The same 16x16 world that defeated the table "
          f"({wall.count_states(env):,} situations).")
    print("The brain has 4,800 weights — it COULD NOT memorize them all")
    print("even if it wanted to. It is forced to find the pattern.\n")

    def net_value(state, action):
        return agent.q_values(env.encode(state))[action]

    for ep in agent.train(env, 1200, report_every=300):
        rate = wall.success_rate(net_value, OpenWorld(size=16, seed=99),
                                 n_episodes=100)
        print(f"  after {ep:4d} episodes: success on fresh exams {rate:.0%}")

    print("\nRecall the wall: the table scored ~18% after 4000 episodes.")
    print("The brain scores ~100% after 1200 — on situations it has never")
    print("seen — because it stopped memorizing answers and learned the idea.")
    print("\nThat is generalization. You just watched it happen.")
