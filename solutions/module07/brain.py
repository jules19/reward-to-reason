"""Module 7 solution — The Brain."""

import random

import numpy as np

from r2r.tensor import Tensor, MLP, Adam


class DeepQ:
    """Q-learning where the table is replaced by a function.

    The update rule is EXACTLY Module 4's:

        target = r + gamma * max_a' Q(s', a')

    The only change is what we do with the target. A table would store it.
    We instead nudge a neural network's weights so its output moves toward
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
        return list(self.net(np.array([encoded_state])).data[0])

    def choose(self, encoded_state):
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_actions)
        return int(np.argmax(self.q_values(encoded_state)))

    def remember(self, state, action, reward, next_state, terminal):
        """Store one experience. `terminal` means next_state has no future
        (reaching the goal — NOT running out of time)."""
        self.buffer.append((state, action, reward, next_state, terminal))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def train_step(self):
        """One nudge of the weights, from one random batch of memories.

        Replaying SHUFFLED old memories (instead of only the last step)
        keeps the network from obsessing over whatever just happened.
        """
        if len(self.buffer) < max(self.warmup, self.batch_size):
            return
        batch = self._rng.sample(self.buffer, self.batch_size)
        states = np.array([b[0] for b in batch])
        actions = [b[1] for b in batch]
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        terminal = np.array([float(b[4]) for b in batch])

        # The Module 4 target, computed for the whole batch at once.
        next_best = self.net(next_states).data.max(axis=1)
        targets = rewards + self.gamma * next_best * (1.0 - terminal)

        # Loss: squared error between Q(s, a) and target, for taken actions.
        out = self.net(states)
        mask = np.zeros((self.batch_size, self.n_actions))
        mask[np.arange(self.batch_size), actions] = 1.0
        diff = out * Tensor(mask) - Tensor(mask * targets[:, None])
        loss = (diff * diff).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run_episode(self, env):
        state = env.reset()
        done = False
        total = 0.0
        while not done:
            encoded = env.encode(state)
            action = self.choose(encoded)
            next_state, reward, done = env.step(action)
            self.remember(encoded, action, reward, env.encode(next_state),
                          done and reward > 0)
            self.train_step()
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_step)
            state = next_state
            total += reward
        return total

    def train(self, env, episodes, report_every=None):
        for ep in range(episodes):
            self.run_episode(env)
            if report_every and (ep + 1) % report_every == 0:
                yield ep + 1


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from module05.the_wall import success_rate, count_states
    from r2r.envs import OpenWorld

    env = OpenWorld(size=16, seed=7)
    agent = DeepQ(state_size=4, n_actions=env.n_actions, seed=0)

    print(f"The same 16x16 world that defeated the table "
          f"({count_states(env):,} situations).")
    print("The brain has 4,800 weights — it COULD NOT memorize them all")
    print("even if it wanted to. It is forced to find the pattern.\n")

    def net_value(state, action):
        return agent.q_values(env.encode(state))[action]

    for ep in agent.train(env, 1200, report_every=300):
        rate = success_rate(net_value, OpenWorld(size=16, seed=99), n_episodes=100)
        print(f"  after {ep:4d} episodes: success on fresh exams {rate:.0%}")

    print("\nRecall the wall: the table scored ~16% after 4000 episodes.")
    print("The brain scores ~100% after 1200 — on situations it has never")
    print("seen — because it stopped memorizing answers and learned the idea.")
    print("\nThat is generalization. You just watched it happen.")
