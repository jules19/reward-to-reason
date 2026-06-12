"""Module 10: Reason. (The capstone — a couple of minutes of self-play.)"""

import numpy as np
import pytest

from r2r.arena import play_match
from r2r.builtin import MinimaxAgent, RandomAgent
from r2r.course import load
from r2r.envs import TicTacToe


def test_predict_speaks_proper_probabilities():
    mod = load(10, "reason")
    game = TicTacToe()
    net = mod.PolicyValueNet(game, seed=0)
    state = game.next_state(game.initial_state(), 4)
    priors, value = net.predict(state)
    assert set(priors) == set(game.legal_moves(state)), \
        "priors must cover exactly the legal moves"
    assert abs(sum(priors.values()) - 1.0) < 1e-9
    assert all(p > 0 for p in priors.values())
    assert -1.0 <= value <= 1.0


def test_training_fits_known_targets():
    mod = load(10, "reason")
    game = TicTacToe()
    net = mod.PolicyValueNet(game, lr=3e-3, seed=0)
    s0 = game.initial_state()
    s1 = game.next_state(s0, 0)
    states = [game.encode(s0), game.encode(s1)]
    policy = np.zeros((2, 9))
    policy[0, 4] = 1.0   # "always take the centre"
    policy[1, 4] = 1.0
    values = np.array([0.9, -0.9])
    for _ in range(150):
        net.train_batch(states, policy, values)
    priors0, value0 = net.predict(s0)
    _, value1 = net.predict(s1)
    assert max(priors0, key=priors0.get) == 4, \
        "after training, the policy head should prefer the taught move"
    assert value0 > 0.5 and value1 < -0.5, \
        "the value head should reproduce the taught judgements"


def test_guided_search_finds_a_win_despite_clueless_intuition():
    """An UNTRAINED net guides the search — terminal truth must still win."""
    mod = load(10, "reason")
    game = TicTacToe()
    net = mod.PolicyValueNet(game, seed=1)
    mcts = mod.GuidedMCTS(net, n_simulations=300, root_noise=0.0, seed=0)
    state = ((1, 1, 0, -1, -1, 0, 0, 0, 0), 1)
    assert mcts.select_move(game, state) == 2


def test_self_play_produces_honest_training_data():
    mod = load(10, "reason")
    game = TicTacToe()
    engine = mod.AlphaLite(game, n_simulations=25, seed=0)
    engine.self_play_game()
    assert len(engine.buffer) >= 5, "every position of the game becomes data"
    for encoded, policy, z in engine.buffer:
        assert len(encoded) == game.encoded_size
        assert abs(sum(policy) - 1.0) < 1e-9
        assert z in (-1.0, 0.0, 1.0)


@pytest.fixture(scope="module")
def trained_engine():
    mod = load(10, "reason")
    game = TicTacToe()
    engine = mod.AlphaLite(game, hidden=64, n_simulations=50,
                           temperature_moves=4, buffer_size=3000, seed=0)
    for _ in range(25):
        engine.train_iteration(n_games=10, n_batches=30)
    return game, engine


@pytest.mark.slow
def test_the_loop_creates_strength(trained_engine):
    game, engine = trained_engine
    agent = engine.player(n_simulations=100)
    w, l, d = play_match(game, agent, RandomAgent(seed=1), n_games=60)
    assert l <= 2, f"after 250 self-play games it lost {l}/60 to RANDOM"
    assert w >= 45


@pytest.mark.slow
def test_the_student_surpasses_the_blind_dreamer(trained_engine):
    """Intuition + imagination beats blind imagination at the same budget."""
    game, engine = trained_engine
    MCTS = load(9, "imagination").MCTS
    w, l, d = play_match(game, engine.player(n_simulations=50),
                         MCTS(n_simulations=50, seed=3), n_games=60)
    assert w > l, f"guided search should outplay raw MCTS ({w} wins, {l} losses)"


@pytest.mark.slow
def test_a_perfect_player_cannot_beat_it(trained_engine):
    game, engine = trained_engine
    agent = engine.player(n_simulations=800)
    w, l, d = play_match(game, agent, MinimaxAgent(randomize=7), n_games=30)
    assert l <= 1, (
        f"with deep search the engine should hold perfect play to draws "
        f"(lost {l}/30)"
    )
