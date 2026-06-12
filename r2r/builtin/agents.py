"""Built-in reference agents ("built-in chips").

These exist so that demos, evaluations and the Module 0 boss fight work
before you have built anything. You will build your own versions of every
one of these during the course — and the tests pit YOUR versions against
the same environments, never against this file's internals.
"""

import math
import random


class RandomAgent:
    """The floor. Every agent you build must climb past this one."""

    def __init__(self, seed=None):
        self._rng = random.Random(seed)

    def select_move(self, game, state):
        return self._rng.choice(game.legal_moves(state))


class MinimaxAgent:
    """Perfect play by exhaustive search. Only viable on tiny games.

    Used as the final exam for tic-tac-toe agents: nothing should ever
    beat it; a well-trained agent should always draw it. Pass a seed to
    `randomize` to make it choose uniformly among equally-perfect moves —
    a much harsher examiner, because it drags opponents into every corner
    of the game tree.
    """

    def __init__(self, randomize=None):
        self._cache = {}
        self._rng = random.Random(randomize) if randomize is not None else None

    def select_move(self, game, state):
        _, moves = self._value(game, state)
        if self._rng is not None:
            return self._rng.choice(moves)
        return moves[0]

    def _value(self, game, state):
        if state in self._cache:
            return self._cache[state]
        w = game.winner(state)
        if w is not None:
            return (w, [])
        player = game.current_player(state)
        best_value, best_moves = None, []
        for move in game.legal_moves(state):
            value, _ = self._value(game, game.next_state(state, move))
            if best_value is None or value * player > best_value * player:
                best_value, best_moves = value, [move]
            elif value == best_value:
                best_moves.append(move)
        self._cache[state] = (best_value, best_moves)
        return best_value, best_moves


class MCTSAgent:
    """Monte Carlo Tree Search with random rollouts (UCT).

    Pure imagination, zero learning: it plays out thousands of random
    futures and follows the statistics. With enough simulations this is
    the Module 0 boss — strong enough to beat most humans at Connect Four
    while containing not one line of game-specific knowledge.
    """

    def __init__(self, n_simulations=400, c_uct=1.4, seed=None):
        self.n_simulations = n_simulations
        self.c_uct = c_uct
        self._rng = random.Random(seed)

    def select_move(self, game, state):
        visits = {}   # (state, move) -> visit count
        totals = {}   # (state, move) -> sum of results for the mover
        for _ in range(self.n_simulations):
            self._simulate(game, state, visits, totals)
        moves = game.legal_moves(state)
        return max(moves, key=lambda m: visits.get((state, m), 0))

    def _simulate(self, game, state, visits, totals):
        path = []
        # Selection / expansion: walk the tree by UCT until we leave it.
        while game.winner(state) is None:
            moves = game.legal_moves(state)
            unexplored = [m for m in moves if (state, m) not in visits]
            if unexplored:
                move = self._rng.choice(unexplored)
                path.append((state, move))
                state = game.next_state(state, move)
                break
            log_n = math.log(sum(visits[(state, m)] for m in moves))
            move = max(moves, key=lambda m: self._uct(state, m, visits, totals, log_n))
            path.append((state, move))
            state = game.next_state(state, move)
        # Rollout: play randomly to the end.
        result = self._rollout(game, state)
        # Backpropagation: credit every move on the path, from its mover's view.
        for s, m in path:
            mover = game.current_player(s)
            visits[(s, m)] = visits.get((s, m), 0) + 1
            totals[(s, m)] = totals.get((s, m), 0.0) + result * mover

    def _uct(self, state, move, visits, totals, log_n):
        n = visits[(state, move)]
        return totals[(state, move)] / n + self.c_uct * math.sqrt(log_n / n)

    def _rollout(self, game, state):
        while True:
            w = game.winner(state)
            if w is not None:
                return w
            state = game.next_state(state, self._rng.choice(game.legal_moves(state)))
