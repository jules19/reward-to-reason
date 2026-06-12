"""Module 3 — A Place to Stand.

Give the learner a memory of PLACES, and watch a map condense out of
nothing but wandering and one distant reward.

Read README.md first.
Check your work:   pytest tests/test_module03.py
See it live:       python modules/03-a-place-to-stand/experience_map.py
"""

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
        """Current estimate for (state, action); 0.0 if never seen."""
        # TODO 1: look it up in self.q (dict.get with a default is enough).
        raise NotImplementedError("TODO 1")

    def choose(self, state):
        """Epsilon-greedy, exactly like Module 2 — but the estimates now
        depend on WHERE YOU ARE. Ties broken randomly."""
        # TODO 2
        raise NotImplementedError("TODO 2")

    def learn_from_episode(self, trajectory, final_reward):
        """trajectory is the list of (state, action) pairs, in order.

        For the step taken t-from-the-end, the target is

            target = final_reward * discount ** (steps_remaining_after_t)

        i.e. the LAST step before the goal deserves nearly full credit,
        and credit fades by `discount` for every step further back.
        Then nudge, with Module 1's rule:

            q[s, a] <- q[s, a] + learning_rate * (target - q[s, a])
        """
        # TODO 3: loop over the trajectory and apply the rule.
        # (If trajectory has length T, the step at index t has
        #  T - 1 - t steps after it.)
        raise NotImplementedError("TODO 3")

    def run_episode(self, env):
        """One life in the maze: act, remember the trajectory, learn at
        the end. Returns the final reward."""
        # TODO 4:
        #   state = env.reset()
        #   loop: choose an action, step the env, record (state, action)
        #         pairs in a list, until done
        #   call learn_from_episode, return the final reward
        raise NotImplementedError("TODO 4")

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
