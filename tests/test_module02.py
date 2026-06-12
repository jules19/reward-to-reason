"""Module 2: The Greedy Trap."""

from r2r.course import load
from r2r.envs import Casino


def test_greedy_exploits():
    mod = load(2, "explorer")
    agent = mod.GreedyAgent(4, seed=0)
    agent.estimates = [0.0, 0.5, 0.2, 0.1]
    assert all(agent.choose() == 1 for _ in range(20))


def test_greedy_breaks_ties_randomly():
    mod = load(2, "explorer")
    agent = mod.GreedyAgent(4, seed=0)
    assert {agent.choose() for _ in range(200)} == {0, 1, 2, 3}, \
        "with all-equal estimates, every action should sometimes be chosen"


def test_epsilon_greedy_doubts():
    mod = load(2, "explorer")
    agent = mod.EpsilonGreedyAgent(4, epsilon=0.5, seed=0)
    agent.estimates = [0.0, 0.5, 0.2, 0.1]
    choices = [agent.choose() for _ in range(400)]
    assert {0, 2, 3} & set(choices), "epsilon must sometimes pick non-best actions"
    assert choices.count(1) > 150, "but mostly it should exploit"


def test_doubt_beats_certainty():
    mod = load(2, "explorer")
    greedy_found = explorer_found = 0
    n_runs = 100
    for seed in range(n_runs):
        env = Casino(seed=seed)
        greedy = mod.GreedyAgent(env.n_actions, seed=seed)
        mod.lifetime_reward(env, greedy, n_pulls=600)
        env = Casino(seed=seed)
        explorer = mod.EpsilonGreedyAgent(env.n_actions, epsilon=0.1, seed=seed)
        mod.lifetime_reward(env, explorer, n_pulls=600)
        best = env.best_action
        greedy_found += max(range(10), key=lambda i: greedy.estimates[i]) == best
        explorer_found += max(range(10), key=lambda i: explorer.estimates[i]) == best
    assert explorer_found / n_runs >= 0.65, \
        "an epsilon-greedy agent should usually locate the best machine"
    assert explorer_found - greedy_found >= 10, \
        "exploration should clearly beat pure greed at finding the truth"
