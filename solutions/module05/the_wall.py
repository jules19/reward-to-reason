"""Module 5 solution — The Wall.

There is nothing to build here except instruments. This module is a
crime-scene investigation: your Module 4 learner is about to fail, and
your job is to measure exactly how and why.
"""

import random


def count_states(env):
    """How many distinct (agent, goal) situations exist in an OpenWorld."""
    cells = env.size * env.size
    return cells * (cells - 1)


def coverage(q_table, env):
    """What fraction of the world's states has the agent EVER seen?

    q_table is a dict keyed by (state, action).
    """
    seen = {state for (state, _action) in q_table}
    return len(seen) / count_states(env)


def success_rate(value_fn, env, n_episodes=200, seed=123):
    """Run greedy episodes on FRESH situations. The only honest exam.

    value_fn(state, action) -> the agent's estimate. Ties broken randomly.
    """
    rng = random.Random(seed)
    successes = 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            values = [value_fn(state, a) for a in range(env.n_actions)]
            best = max(values)
            action = rng.choice([a for a, v in enumerate(values) if v == best])
            state, reward, done = env.step(action)
        agent, goal = state
        successes += agent == goal
    return successes / n_episodes


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from module04.q_learner import QLearner
    from r2r.envs import OpenWorld

    env = OpenWorld(size=16, seed=7)
    print("A 16x16 room where the goal moves every episode.")
    print(f"Distinct situations: {count_states(env):,}\n")

    agent = QLearner(env.n_actions, seed=0)
    print("Training your Module 4 learner for 4000 episodes "
          "(40x more than the maze needed)...")
    agent.train(env, 4000)

    cov = coverage(agent.q, env)
    rate = success_rate(agent.value, OpenWorld(size=16, seed=99))
    print(f"\n  situations ever visited : {cov:.0%}")
    print(f"  success on fresh exams  : {rate:.0%}\n")
    print("It has seen most of the world and still fails almost every test.")
    print("A table can only remember. Every cell of it is a private universe:")
    print("knowing the way from (3,4) to (9,9) teaches it NOTHING about the")
    print("way from (3,5) to (9,9). It cannot notice that the two are almost")
    print("the same problem. It cannot generalize.")
    print("\nNo table will ever play chess. We need a different kind of memory —")
    print("one that learns the IDEA of 'walk toward the goal'.")
    print("That requires a machine that can follow blame backward.")
    print("Proceed to Module 6.")
