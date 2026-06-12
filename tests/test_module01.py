"""Module 1: The Coin That Learns."""

from r2r.course import load
from r2r.envs import CoinFlip


def make_agent(**kwargs):
    return load(1, "coin_agent").LearningCoin(**kwargs)


def test_starts_ignorant():
    agent = make_agent(seed=0)
    assert agent.estimates == [0.0, 0.0]


def test_nudge_rule_exact():
    agent = make_agent(learning_rate=0.1, seed=0)
    agent.learn(0, 1.0)
    assert abs(agent.estimates[0] - 0.1) < 1e-12, \
        "one reward of 1.0 from estimate 0.0 should give exactly lr * 1.0"
    agent.learn(0, 1.0)
    assert abs(agent.estimates[0] - 0.19) < 1e-12, \
        "the nudge must shrink as the estimate approaches the reward"
    assert agent.estimates[1] == 0.0, "only the taken action may learn"


def test_explore_reaches_both_actions():
    agent = make_agent(seed=3)
    seen = {agent.explore() for _ in range(100)}
    assert seen == {0, 1}


def test_best_guess_reads_estimates():
    agent = make_agent(seed=0)
    agent.estimates = [0.4, 0.1]
    assert agent.best_guess() == 0
    agent.estimates = [0.1, 0.9]
    assert agent.best_guess() == 1


def test_coin_discovers_the_truth():
    mod = load(1, "coin_agent")
    env = CoinFlip(seed=42)
    agent = mod.LearningCoin(seed=7)
    estimates = mod.run_experiment(env, agent, n_pulls=500)
    assert agent.best_guess() == env.best_action, \
        "after 500 flips the coin should know which action is better"
    assert abs(estimates[1] - 0.8) < 0.25
    assert abs(estimates[0] - 0.2) < 0.25
