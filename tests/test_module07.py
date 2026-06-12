"""Module 7: The Brain. (The slowest test file — about a minute of
training. Generalization has to be earned.)"""

import random

from r2r.course import load
from r2r.envs import OpenWorld


def make_agent(**kwargs):
    defaults = dict(state_size=4, n_actions=4, seed=0)
    defaults.update(kwargs)
    return load(7, "brain").DeepQ(**defaults)


def test_q_values_shape():
    agent = make_agent()
    q = agent.q_values([0.1, 0.2, 0.3, 0.4])
    assert len(list(q)) == 4


def test_choose_exploits_when_certain():
    agent = make_agent()
    agent.epsilon = 0.0
    state = [0.5, 0.5, 0.0, 0.0]
    expected = max(range(4), key=lambda a: agent.q_values(state)[a])
    assert all(agent.choose(state) == expected for _ in range(5))


def test_choose_explores_when_doubtful():
    agent = make_agent()
    agent.epsilon = 1.0
    assert {agent.choose([0.0] * 4) for _ in range(100)} == {0, 1, 2, 3}


def test_memory_is_bounded():
    agent = make_agent(buffer_size=10)
    for i in range(25):
        agent.remember([i] * 4, 0, 0.0, [i] * 4, False)
    assert len(agent.buffer) == 10
    assert agent.buffer[0][0] == [15] * 4, "oldest memories must be dropped first"


def test_training_changes_the_weights():
    agent = make_agent(warmup=8, batch_size=8)
    rng = random.Random(0)
    for _ in range(20):
        s = [rng.random() for _ in range(4)]
        agent.remember(s, rng.randrange(4), rng.random(), s, False)
    before = [p.data.copy() for p in agent.net.parameters()]
    agent.train_step()
    moved = any(abs(p.data - b).max() > 0 for p, b in zip(agent.net.parameters(), before))
    assert moved, "train_step must actually nudge the network"


def test_the_brain_beats_the_wall():
    """The table scored ~16% after 4000 episodes (Module 5). The brain
    must crush that score with 600."""
    env = OpenWorld(size=16, seed=7)
    agent = make_agent(seed=0)
    for _ in range(600):
        agent.run_episode(env)
    wall = load(5, "the_wall")

    def net_value(state, action):
        return agent.q_values(env.encode(state))[action]

    rate = wall.success_rate(net_value, OpenWorld(size=16, seed=99), n_episodes=100)
    assert rate >= 0.65, (
        f"success on fresh situations was {rate:.0%}; the brain should "
        "generalize to at least 65% (a typical run reaches 95%+ — the "
        "table managed 16% with 6x the experience)"
    )
