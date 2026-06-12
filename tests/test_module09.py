"""Module 9: Imagination."""

from r2r.arena import play_match
from r2r.builtin import MinimaxAgent, RandomAgent
from r2r.course import load
from r2r.envs import ConnectFour, TicTacToe


def test_minimax_takes_an_immediate_win():
    Minimax = load(9, "imagination").Minimax
    game = TicTacToe()
    state = ((1, 1, 0, -1, -1, 0, 0, 0, 0), 1)
    assert Minimax().select_move(game, state) == 2


def test_minimax_blocks_a_threat():
    Minimax = load(9, "imagination").Minimax
    game = TicTacToe()
    # O to move; X threatens to complete the top row at square 2.
    state = ((1, 1, 0, 0, -1, 0, 0, 0, 0), -1)
    assert Minimax().select_move(game, state) == 2


def test_minimax_is_perfect():
    Minimax = load(9, "imagination").Minimax
    game = TicTacToe()
    agent = Minimax()
    # Perfect vs perfect is always a draw...
    w, l, d = play_match(game, agent, MinimaxAgent(randomize=3), n_games=20)
    assert l == 0 and w == 0 and d == 20
    # ...and perfect vs random never loses.
    w, l, d = play_match(game, agent, RandomAgent(seed=2), n_games=50)
    assert l == 0


def test_mcts_takes_an_immediate_win():
    MCTS = load(9, "imagination").MCTS
    game = ConnectFour()
    state = game.initial_state()
    for move in (0, 6, 0, 6, 0, 5):   # X has three stacked in column 0
        state = game.next_state(state, move)
    assert MCTS(n_simulations=400, seed=0).select_move(game, state) == 0


def test_mcts_blocks_an_immediate_loss():
    MCTS = load(9, "imagination").MCTS
    game = ConnectFour()
    state = game.initial_state()
    for move in (0, 3, 0, 4, 6, 5):   # O threatens 3-4-5-(6 or 2) on the floor
        state = game.next_state(state, move)
    move = MCTS(n_simulations=600, seed=1).select_move(game, state)
    assert move == 2, "O completes 2-3-4-5 unless X blocks at column 2"


def test_mcts_crushes_random_play():
    MCTS = load(9, "imagination").MCTS
    game = ConnectFour()
    w, l, d = play_match(game, MCTS(n_simulations=150, seed=0),
                         RandomAgent(seed=1), n_games=20)
    assert w >= 18, f"150 imagined futures per move should win ~always ({w}/20)"


def test_deeper_imagination_wins():
    MCTS = load(9, "imagination").MCTS
    game = ConnectFour()
    w, l, d = play_match(game, MCTS(n_simulations=400, seed=0),
                         MCTS(n_simulations=25, seed=1), n_games=12)
    assert w > l, "an agent that imagines 16x more futures should dominate"
