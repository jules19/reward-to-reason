"""Module 3: A Place to Stand."""

from r2r.course import load
from r2r.envs import Maze, SMALL_MAZE


def make_agent(**kwargs):
    return load(3, "experience_map").ExperienceMap(4, **kwargs)


def test_value_defaults_to_zero():
    agent = make_agent(seed=0)
    assert agent.value(("anywhere",), 2) == 0.0


def test_credit_fades_with_distance():
    agent = make_agent(learning_rate=0.2, discount=0.95, seed=0)
    trajectory = [("s1", 0), ("s2", 1)]
    agent.learn_from_episode(trajectory, 1.0)
    assert abs(agent.value("s2", 1) - 0.2) < 1e-12, \
        "the last step before the reward gets the full (nudged) credit"
    assert abs(agent.value("s1", 0) - 0.19) < 1e-12, \
        "one step earlier the credit fades by the discount: 0.2 * 0.95"


def test_failed_episodes_teach_too():
    agent = make_agent(learning_rate=0.5, seed=0)
    agent.q[("s1", 0)] = 0.8
    agent.learn_from_episode([("s1", 0)], 0.0)
    assert abs(agent.value("s1", 0) - 0.4) < 1e-12, \
        "an episode that ends with no reward should DECAY old optimism"


def test_choose_is_epsilon_greedy():
    agent = make_agent(epsilon=0.0, seed=0)
    agent.q[("s", 2)] = 1.0
    assert all(agent.choose("s") == 2 for _ in range(10))
    agent.epsilon = 1.0
    assert {agent.choose("s") for _ in range(100)} == {0, 1, 2, 3}


def test_solves_the_small_maze():
    env = Maze(SMALL_MAZE)
    agent = make_agent(learning_rate=0.1, epsilon=0.2, seed=0)
    agent.train(env, 600)
    path = agent.greedy_path(env)
    assert path[-1] == env.goal, "after 600 episodes the greedy path should reach G"
    assert len(path) - 1 <= 24, "and it should be a reasonably direct route"
