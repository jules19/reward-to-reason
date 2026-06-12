"""Module 3 solution — A Place to Stand."""

import random


class ExperienceMap:
    """A learner with a memory of places.

    q[(state, action)] estimates "how good is doing `action` in `state`?".
    It learns the slow, honest way: live a whole episode, and if it ends
    in reward, walk back through the memory crediting every step — with
    the credit fading the further it was from the moment of success.
    """

    def __init__(self, n_actions, learning_rate=0.2, epsilon=0.2,
                 discount=0.95, seed=None):
        self.q = {}
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount = discount
        self._rng = random.Random(seed)

    def value(self, state, action):
        return self.q.get((state, action), 0.0)

    def choose(self, state):
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_actions)
        values = [self.value(state, a) for a in range(self.n_actions)]
        best = max(values)
        candidates = [a for a, v in enumerate(values) if v == best]
        return self._rng.choice(candidates)

    def learn_from_episode(self, trajectory, final_reward):
        """trajectory is the list of (state, action) pairs, in order."""
        T = len(trajectory)
        for t, (state, action) in enumerate(trajectory):
            target = final_reward * self.discount ** (T - 1 - t)
            old = self.value(state, action)
            self.q[(state, action)] = old + self.learning_rate * (target - old)

    def run_episode(self, env):
        """One life in the maze. Returns the final reward."""
        state = env.reset()
        trajectory = []
        done = False
        reward = 0.0
        while not done:
            action = self.choose(state)
            trajectory.append((state, action))
            state, reward, done = env.step(action)
        self.learn_from_episode(trajectory, reward)
        return reward

    def train(self, env, episodes):
        for _ in range(episodes):
            self.run_episode(env)

    def greedy_path(self, env, max_steps=200):
        """Follow current beliefs with no exploration. Returns visited states."""
        state = env.reset()
        path = [state]
        for _ in range(max_steps):
            values = [self.value(state, a) for a in range(self.n_actions)]
            action = max(range(self.n_actions), key=lambda a: values[a])
            state, reward, done = env.step(action)
            path.append(state)
            if done:
                break
        return path


if __name__ == "__main__":
    from r2r.envs import Maze, SMALL_MAZE

    env = Maze(SMALL_MAZE)
    agent = ExperienceMap(env.n_actions, learning_rate=0.1, seed=0)
    print("Training in the maze (the agent sees only walls it bumps into)...\n")
    for batch in range(10):
        agent.train(env, 50)
        path = agent.greedy_path(env)
        solved = path[-1] == env.goal
        print(f"after {50 * (batch + 1):3d} episodes: greedy path "
              f"{'reaches the goal in ' + str(len(path) - 1) + ' steps' if solved else 'gets lost'}")
    print()
    path = agent.greedy_path(env)
    overlay = {p: "o" for p in path}
    overlay[path[0]], overlay[path[-1]] = "S", "G"
    for r, row in enumerate(env.grid):
        print("".join(overlay.get((r, c), ch) for c, ch in enumerate(row)))
    print("\nNobody drew it a map. The map condensed out of experience.")
