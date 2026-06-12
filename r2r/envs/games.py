"""Two-player, perfect-information games — the worlds with an opponent.

All games share one interface (the `Game` class), so every agent built in
Acts II and III — tabular learners, minimax, MCTS, AlphaZero-lite — is
game-agnostic. Swap TicTacToe for ConnectFour and nothing else changes.

Conventions:
  * Players are +1 and -1. Player +1 always moves first.
  * A state is an immutable (board, player_to_move) tuple, so it can be
    used as a dictionary key.
  * `winner(state)` returns +1 or -1 for a decided game, 0 for a draw,
    and None while the game is still in progress.
  * `encode(state)` returns a flat list of floats from the perspective of
    the player to move (own pieces first), so a single network can play
    both colours.
"""


class Game:
    """The interface every game implements."""

    name = "game"
    n_actions = 0       # size of the (fixed) action space
    encoded_size = 0    # length of the list returned by encode()

    def initial_state(self):
        raise NotImplementedError

    def legal_moves(self, state):
        raise NotImplementedError

    def next_state(self, state, move):
        raise NotImplementedError

    def winner(self, state):
        raise NotImplementedError

    def current_player(self, state):
        return state[1]

    def encode(self, state):
        raise NotImplementedError

    def render(self, state):
        raise NotImplementedError


class TicTacToe(Game):
    """3x3 noughts and crosses. Small enough to solve, rich enough to learn."""

    name = "tictactoe"
    n_actions = 9
    encoded_size = 18

    _LINES = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
        (0, 4, 8), (2, 4, 6),              # diagonals
    ]

    def initial_state(self):
        return ((0,) * 9, 1)

    def legal_moves(self, state):
        board, _ = state
        return [i for i in range(9) if board[i] == 0]

    def next_state(self, state, move):
        board, player = state
        if board[move] != 0:
            raise ValueError(f"illegal move {move}")
        new_board = list(board)
        new_board[move] = player
        return (tuple(new_board), -player)

    def winner(self, state):
        board, _ = state
        for a, b, c in self._LINES:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return board[a]
        if 0 not in board:
            return 0
        return None

    def encode(self, state):
        board, player = state
        mine = [1.0 if v == player else 0.0 for v in board]
        theirs = [1.0 if v == -player else 0.0 for v in board]
        return mine + theirs

    def render(self, state):
        board, player = state
        sym = {1: "X", -1: "O", 0: "."}
        rows = []
        for r in range(3):
            rows.append(" ".join(sym[board[3 * r + c]] for c in range(3)))
        return "\n".join(rows) + f"\n({sym[player]} to move)"


class ConnectFour(Game):
    """6x7 Connect Four. Big enough that brute force dies and reason begins.

    The board tuple is row-major with row 0 at the TOP. A move is a column
    index 0-6; the piece falls to the lowest empty cell in that column.
    """

    name = "connect4"
    n_actions = 7
    encoded_size = 84
    ROWS, COLS = 6, 7

    def initial_state(self):
        return ((0,) * (self.ROWS * self.COLS), 1)

    def legal_moves(self, state):
        board, _ = state
        return [c for c in range(self.COLS) if board[c] == 0]

    def next_state(self, state, move):
        board, player = state
        if board[move] != 0:
            raise ValueError(f"illegal move {move}: column {move} is full")
        new_board = list(board)
        for r in range(self.ROWS - 1, -1, -1):
            i = r * self.COLS + move
            if new_board[i] == 0:
                new_board[i] = player
                break
        return (tuple(new_board), -player)

    def winner(self, state):
        board, _ = state
        R, C = self.ROWS, self.COLS
        for r in range(R):
            for c in range(C):
                v = board[r * C + c]
                if v == 0:
                    continue
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    rr, cc = r + 3 * dr, c + 3 * dc
                    if not (0 <= rr < R and 0 <= cc < C):
                        continue
                    if all(board[(r + k * dr) * C + (c + k * dc)] == v for k in range(1, 4)):
                        return v
        if 0 not in board:
            return 0
        return None

    def encode(self, state):
        board, player = state
        mine = [1.0 if v == player else 0.0 for v in board]
        theirs = [1.0 if v == -player else 0.0 for v in board]
        return mine + theirs

    def render(self, state):
        board, player = state
        sym = {1: "X", -1: "O", 0: "."}
        rows = [" ".join(str(c) for c in range(self.COLS))]
        for r in range(self.ROWS):
            rows.append(" ".join(sym[board[r * self.COLS + c]] for c in range(self.COLS)))
        return "\n".join(rows) + f"\n({sym[player]} to move)"


def play_game(game, agent_plus, agent_minus, render=False):
    """Play one game between two agents. Returns +1, -1 or 0.

    An agent is anything with a `select_move(game, state)` method.
    """
    state = game.initial_state()
    agents = {1: agent_plus, -1: agent_minus}
    while game.winner(state) is None:
        player = game.current_player(state)
        move = agents[player].select_move(game, state)
        state = game.next_state(state, move)
        if render:
            print(game.render(state) + "\n")
    return game.winner(state)
