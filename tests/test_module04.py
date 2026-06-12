"""Module 4: Messages From the Future."""

from r2r.course import load
from r2r.envs import Maze, SMALL_MAZE
from r2r.envs.gridworld import Cliff


def make_agent(**kwargs):
    return load(4, "q_learner").QLearner(4, **kwargs)


def test_update_uses_the_next_states_best_future():
    agent = make_agent(learning_rate=0.5, discount=0.9, seed=0)
    agent.q[("s2", 1)] = 0.6   # the best the agent believes about s2
    agent.q[("s2", 0)] = 0.2
    agent.learn("s1", 3, 0.0, "s2", done=False)
    # target = 0 + 0.9 * 0.6 = 0.54; q <- 0 + 0.5 * 0.54
    assert abs(agent.value("s1", 3) - 0.27) < 1e-12


def test_done_means_no_future():
    agent = make_agent(learning_rate=0.5, discount=0.9, seed=0)
    agent.q[("after",) * 1] = 99.0  # must be ignored
    agent.learn("s1", 0, 1.0, "terminal", done=True)
    assert abs(agent.value("s1", 0) - 0.5) < 1e-12, \
        "when done, the target is the reward alone"


def test_learns_online_without_trajectories():
    """The whole point: a single transition teaches, no episode needed."""
    agent = make_agent(learning_rate=1.0, discount=0.5, seed=0)
    agent.learn("a", 0, 0.0, "b", done=False)
    agent.learn("b", 0, 1.0, "end", done=True)
    agent.learn("a", 0, 0.0, "b", done=False)
    assert agent.value("a", 0) > 0.0, \
        "value must flow backward from b to a through the bootstrap"


def test_solves_the_small_maze():
    env = Maze(SMALL_MAZE)
    agent = make_agent(seed=0)
    agent.train(env, 500)
    path = agent.greedy_path(env)
    assert path[-1] == env.goal
    assert len(path) - 1 <= 24


def test_walks_the_cliff_edge():
    env = Cliff()
    agent = make_agent(learning_rate=0.2, epsilon=0.1, seed=0)
    agent.train(env, 5000)
    state = env.reset()
    path = [state]
    for _ in range(50):
        action = max(range(4), key=lambda a: agent.value(state, a))
        state, reward, done = env.step(action)
        path.append(state)
        if done:
            break
    assert path[-1] == env.goal, "greedy path must reach the goal"
    assert len(path) - 1 <= 15, \
        "off-policy learning should find the short, brave route (optimal is 13)"
