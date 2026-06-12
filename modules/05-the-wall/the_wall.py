"""Module 5 — The Wall.

Nothing to build here except instruments. This module is a crime-scene
investigation: your Module 4 learner is about to fail, and your job is
to measure exactly how and why.

Read README.md first.
Check your work:   pytest tests/test_module05.py
Watch it fail:     python modules/05-the-wall/the_wall.py
"""

import random


def count_states(env):
    """How many distinct (agent, goal) situations exist in an OpenWorld?

    The agent can stand on any of the size*size cells, and the goal on
    any OTHER cell.
    """
    # TODO 1
    raise NotImplementedError("TODO 1")


def coverage(q_table, env):
    """What fraction of the world's states has the agent EVER seen?

    q_table is a dict keyed by (state, action) — your QLearner's .q.
    Count distinct STATES that appear in its keys, divide by count_states.
    """
    # TODO 2
    raise NotImplementedError("TODO 2")


def success_rate(value_fn, env, n_episodes=200, seed=123):
    """Run greedy episodes on FRESH situations. The only honest exam.

    value_fn(state, action) -> an estimate. For each episode: reset the
    env, repeatedly take the action with the highest estimate (break ties
    randomly with random.Random(seed)), until done. Count an episode as a
    success if the agent ended ON the goal (state is (agent, goal)).
    Return successes / n_episodes.
    """
    # TODO 3
    raise NotImplementedError("TODO 3")


if __name__ == "__main__":
    from r2r.course import load
    from r2r.envs import OpenWorld

    QLearner = load(4, "q_learner").QLearner

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
