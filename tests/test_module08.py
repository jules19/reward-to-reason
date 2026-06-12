"""Module 8: The Mirror."""

import pytest

from r2r.arena import play_match
from r2r.builtin import MinimaxAgent, RandomAgent
from r2r.envs import TicTacToe


@pytest.fixture(scope="module")
def trained():
    from r2r.course import load
    game = TicTacToe()
    agent = load(8, "mirror").SelfPlayLearner(game, seed=0)
    agent.train(25000)
    return game, agent


def make_agent(**kwargs):
    from r2r.course import load
    return load(8, "mirror").SelfPlayLearner(TicTacToe(), **kwargs)


def test_terminal_positions_are_facts():
    agent = make_agent(seed=0)
    game = agent.game
    # X wins on the top row.
    won = ((1, 1, 1, -1, -1, 0, 0, 0, 0), -1)
    assert agent.value(won) == 1.0
    agent.values[won] = -0.7  # even a corrupted memory must not matter
    assert agent.value(won) == 1.0


def test_best_move_takes_an_immediate_win():
    agent = make_agent(seed=0)
    # X to move, two in a row on top: the winning move is square 2.
    state = ((1, 1, 0, -1, -1, 0, 0, 0, 0), 1)
    assert agent.best_move(state) == 2


def test_best_move_works_for_both_sides_of_the_mirror():
    agent = make_agent(seed=0)
    # O to move, O has two in the left column: winning move is square 6.
    state = ((-1, 1, 1, -1, 1, 0, 0, 0, 0), -1)
    assert agent.best_move(state) == 6, \
        "remember: values are from +1's view, the mover might be -1"


def test_training_game_reaches_a_verdict():
    agent = make_agent(seed=0)
    history = agent.play_training_game()
    assert history[0] == agent.game.initial_state()
    assert agent.game.winner(history[-1]) is not None
    assert all(agent.game.winner(s) is None for s in history[:-1])


def test_learning_nudges_toward_the_result():
    agent = make_agent(learning_rate=0.1, seed=0)
    game = agent.game
    s0 = game.initial_state()
    s1 = game.next_state(s0, 4)
    # A fabricated finished game: X eventually won.
    history = [s0, s1, ((1, 1, 1, -1, -1, 0, 0, 0, 0), -1)]
    agent.learn_from_game(history)
    assert abs(agent.values[s0] - 0.1) < 1e-12
    assert abs(agent.values[s1] - 0.1) < 1e-12


def test_never_loses_to_random(trained):
    game, agent = trained
    w, l, d = play_match(game, agent, RandomAgent(seed=1), n_games=200)
    assert l <= 4, f"lost {l} of 200 games to a RANDOM player after training"
    assert w >= 150


def test_survives_a_perfect_examiner(trained):
    game, agent = trained
    w, l, d = play_match(game, agent, MinimaxAgent(), n_games=2)
    assert l == 0, "a well-trained mirror agent never loses to perfect play"
