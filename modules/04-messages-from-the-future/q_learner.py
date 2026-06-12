"""Module 4 — Messages From the Future.

Stop waiting for the end of the story: learn at every single step, from
your own best beliefs about what comes next.

Read README.md first.
Check your work:   pytest tests/test_module04.py
See it live:       python modules/04-messages-from-the-future/q_learner.py
"""

import random


class QLearner:
    """A learner that does not wait for the end of the story.

    After every single step it updates its belief about (state, action)
    using the best thing it currently believes about the NEXT state:

        q[s, a] <- q[s, a] + lr * (r + gamma * max_a' q[s', a'] - q[s, a])

    Value flows backward from the goal one handshake at a time, episode
    after episode, until the whole maze glows with directions.
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
        # TODO 1: same as Module 3.
        raise NotImplementedError("TODO 1")

    def best_value(self, state):
        """The value of a state if you act as well as you know how:
        max over actions of value(state, action)."""
        # TODO 2
        raise NotImplementedError("TODO 2")

    def choose(self, state):
        # TODO 3: epsilon-greedy with random tie-breaks, same as Module 3.
        raise NotImplementedError("TODO 3")

    def learn(self, state, action, reward, next_state, done):
        """The one-step update. This is the line the whole course pivots on.

            target = reward                          if done
            target = reward + discount * best_value(next_state)  otherwise

            q[s, a] <- q[s, a] + learning_rate * (target - q[s, a])

        Note what is strange here: the target contains your OWN current
        guess about next_state. You are learning from a belief, not a
        fact. (Why this is allowed to work is in the README.)
        """
        # TODO 4
        raise NotImplementedError("TODO 4")

    def run_episode(self, env):
        """One life: act, learn IMMEDIATELY after each step, repeat."""
        # TODO 5: like Module 3's run_episode, but call self.learn(...)
        # inside the loop, right after env.step. No trajectory needed —
        # that's the point. Return total reward.
        raise NotImplementedError("TODO 5")

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
    from r2r.course import load
    from r2r.envs.gridworld import Cliff

    ExperienceMap = load(3, "experience_map").ExperienceMap

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
