"""Module 4 solution — Messages From the Future."""

import random


class QLearner:
    """A learner that does not wait for the end of the story.

    After every single step it updates its belief about (state, action)
    using the best thing it currently believes about the NEXT state:

        q[s, a] <- q[s, a] + lr * (r + gamma * max_a' q[s', a'] - q[s, a])

    Value flows backward from the goal one handshake at a time, episode
    after episode, until the whole maze glows with directions.
    (You invented this. Its official name is Q-learning, and the
    self-consistency idea inside it is the Bellman equation.)
    """

    def __init__(self, n_actions, learning_rate=0.3, epsilon=0.2,
                 discount=0.95, seed=None):
        self.q = {}
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount = discount
        self._rng = random.Random(seed)

    def value(self, state, action):
        return self.q.get((state, action), 0.0)

    def best_value(self, state):
        return max(self.value(state, a) for a in range(self.n_actions))

    def choose(self, state):
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_actions)
        values = [self.value(state, a) for a in range(self.n_actions)]
        best = max(values)
        candidates = [a for a, v in enumerate(values) if v == best]
        return self._rng.choice(candidates)

    def learn(self, state, action, reward, next_state, done):
        """The one-step update. `done` means next_state has no future."""
        future = 0.0 if done else self.discount * self.best_value(next_state)
        old = self.value(state, action)
        self.q[(state, action)] = old + self.learning_rate * (reward + future - old)

    def run_episode(self, env):
        state = env.reset()
        done = False
        total = 0.0
        while not done:
            action = self.choose(state)
            next_state, reward, done = env.step(action)
            self.learn(state, action, reward, next_state, done)
            state = next_state
            total += reward
        return total

    def train(self, env, episodes):
        for _ in range(episodes):
            self.run_episode(env)

    def greedy_path(self, env, max_steps=400):
        state = env.reset()
        path = [state]
        for _ in range(max_steps):
            action = max(range(self.n_actions), key=lambda a: self.value(state, a))
            state, reward, done = env.step(action)
            path.append(state)
            if done:
                break
        return path


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from module03.experience_map import ExperienceMap
    from r2r.envs.gridworld import Cliff

    def greedy_path(agent, env, max_steps=100):
        state = env.reset()
        path = [state]
        for _ in range(max_steps):
            action = max(range(4), key=lambda a: agent.value(state, a))
            state, reward, done = env.step(action)
            path.append(state)
            if done:
                break
        return path

    env = Cliff()
    print("The cliff walk. Fall in (~) and the episode ends at -1.")
    print("Reach G and it ends at +1. The brave path hugs the edge.\n")

    storyteller = ExperienceMap(env.n_actions, seed=0)
    storyteller.train(env, 5000)
    p1 = greedy_path(storyteller, env)

    physicist = QLearner(env.n_actions, learning_rate=0.2, epsilon=0.1, seed=0)
    physicist.train(env, 5000)
    p2 = greedy_path(physicist, env)

    print(f"Module 3 learner (judges actions by how its stories ended): "
          f"{len(p1) - 1} steps")
    print(env.render(p1))
    print(f"\nModule 4 learner (judges actions by the best available future): "
          f"{len(p2) - 1} steps")
    print(env.render(p2))
    print("\nSame world, same reward, same number of lives.")
    print("The story-learner is haunted by its own stumbles near the edge,")
    print("so it learned to fear the fast road. The one-step learner asked a")
    print("different question — not 'what happened to me here?' but 'what is")
    print("the best that can follow from here?' — and walked the edge.")
