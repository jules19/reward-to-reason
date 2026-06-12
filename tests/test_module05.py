"""Module 5: The Wall."""

from r2r.course import load
from r2r.envs import OpenWorld


def test_count_states():
    mod = load(5, "the_wall")
    assert mod.count_states(OpenWorld(size=4)) == 16 * 15
    assert mod.count_states(OpenWorld(size=16)) == 256 * 255


def test_coverage():
    mod = load(5, "the_wall")
    env = OpenWorld(size=4)
    q = {
        ((( 0, 0), (1, 1)), 0): 0.5,
        ((( 0, 0), (1, 1)), 1): 0.2,   # same state, different action
        ((( 2, 2), (3, 3)), 0): 0.1,
    }
    assert abs(mod.coverage(q, env) - 2 / 240) < 1e-12


def test_success_rate_recognizes_competence():
    mod = load(5, "the_wall")

    def smart(state, action):
        """Value an action by how close it lands to the goal."""
        from r2r.envs.gridworld import ACTIONS
        (ar, ac), (gr, gc) = state
        dr, dc = ACTIONS[action]
        nr, nc = min(max(ar + dr, 0), 7), min(max(ac + dc, 0), 7)
        return -(abs(nr - gr) + abs(nc - gc))

    rate = mod.success_rate(smart, OpenWorld(size=8, seed=5), n_episodes=100)
    assert rate >= 0.99, "a perfect navigator must pass the exam"


def test_success_rate_recognizes_ignorance():
    mod = load(5, "the_wall")
    rate = mod.success_rate(lambda s, a: 0.0, OpenWorld(size=8, seed=5),
                            n_episodes=100)
    assert rate <= 0.6, "an ignorant agent (random walk) must not score high"


def test_the_wall_is_real():
    """The heart of the module: a trained table fails fresh exams."""
    mod = load(5, "the_wall")
    QLearner = load(4, "q_learner").QLearner
    env = OpenWorld(size=16, seed=7)
    agent = QLearner(env.n_actions, seed=0)
    agent.train(env, 1500)
    rate = mod.success_rate(agent.value, OpenWorld(size=16, seed=99),
                            n_episodes=100)
    assert rate <= 0.5, (
        "If this fails, your Q-learner somehow beat the wall — "
        "which would be genuinely interesting; check your measurement code."
    )
