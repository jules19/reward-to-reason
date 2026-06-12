"""Module 9 — Imagination.

A completely different road to strength: no learning at all. Just the
ability to imagine futures before choosing.

Read README.md first.
Check your work:   pytest tests/test_module09.py
See it live:       python modules/09-imagination/imagination.py
"""

import math
import random


class Minimax:
    """Perfect play through exhaustive imagination.

    No learning. No memory of past games. It imagines EVERY possible
    future and picks the move with the best guaranteed outcome.
    """

    def __init__(self):
        self._cache = {}

    def select_move(self, game, state):
        _, move = self._search(game, state)
        return move

    def _search(self, game, state):
        """Returns (value, best_move) where value is the game's result
        from +1's point of view assuming BOTH sides play perfectly.

        Recipe:
          * cache hit? return it (game positions repeat — without the
            cache tic-tac-toe takes minutes, with it milliseconds)
          * game over? return (winner, None)
          * otherwise: for each legal move, recurse on the resulting
            state; the current player picks the move maximizing
            player * value. Cache and return.
        """
        # TODO 1
        raise NotImplementedError("TODO 1")


class MCTS:
    """Imagination on a budget — for games too big to imagine fully.

    Where minimax must imagine everything, MCTS imagines a fixed number
    of plausible futures, spending more of its budget on moves that are
    looking good while still occasionally trying doubtful ones. The UCB
    rule below is the casino lesson from Module 2, reborn inside a tree.
    """

    def __init__(self, n_simulations=400, c_explore=1.4, seed=None):
        self.n_simulations = n_simulations
        self.c_explore = c_explore
        self._rng = random.Random(seed)

    def select_move(self, game, state):
        """Run n_simulations from this state, then return the move with
        the most visits (the move it kept wanting to think about)."""
        visits = {}   # (state, move) -> times imagined
        totals = {}   # (state, move) -> sum of results, from the MOVER's view
        for _ in range(self.n_simulations):
            self._simulate(game, state, visits, totals)
        moves = game.legal_moves(state)
        return max(moves, key=lambda m: visits.get((state, m), 0))

    def _simulate(self, game, state, visits, totals):
        """One imagined future, in four phases:

        1. SELECT: while the game isn't over and every legal move from
           `state` has been tried before, descend: pick the move with the
           highest _ucb score, append (state, move) to a path list, step.
        2. EXPAND: the first time you reach a state with untried moves,
           pick one untried move at random, append it to the path, step
           once — then stop descending.
        3. ROLLOUT: from wherever you stopped, _rollout to the end.
        4. BACKPROPAGATE: for every (s, m) on the path:
               visits[(s, m)] += 1
               totals[(s, m)] += result * current_player(s)
           (multiplying by the mover converts the absolute result into
           "was this good for whoever made the move?")
        """
        # TODO 2
        raise NotImplementedError("TODO 2")

    def _ucb(self, state, move, visits, totals, log_n):
        """average result so far  +  c * sqrt(log(parent visits) / visits).

        The first term is greed, the second is curiosity about underexplored
        moves. Module 2, reborn."""
        # TODO 3
        raise NotImplementedError("TODO 3")

    def _rollout(self, game, state):
        """Play random legal moves until the game ends; return the result
        (as a float, from +1's point of view)."""
        # TODO 4
        raise NotImplementedError("TODO 4")


if __name__ == "__main__":
    import time

    from r2r.arena import play_match
    from r2r.builtin import RandomAgent
    from r2r.envs import ConnectFour, TicTacToe

    ttt, c4 = TicTacToe(), ConnectFour()

    print("Two ways to think ahead, zero learning in either.\n")

    t0 = time.time()
    mm = Minimax()
    mm.select_move(ttt, ttt.initial_state())
    print(f"Minimax mapped ALL of tic-tac-toe in {time.time() - t0:.1f}s "
          f"({len(mm._cache):,} positions).")
    print("Connect Four has ~4,500,000,000,000 positions. Minimax is dead.\n")

    print("MCTS on Connect Four, imagining only a few futures per move:")
    for sims in (20, 100, 400):
        agent = MCTS(n_simulations=sims, seed=0)
        w, l, d = play_match(c4, agent, RandomAgent(seed=1), n_games=40)
        print(f"  {sims:4d} dreams/move: w/l/d {w}/{l}/{d} vs random")

    print("\nDeeper dreams beat shallower dreams:")
    w, l, d = play_match(c4, MCTS(800, seed=0), MCTS(50, seed=1), n_games=20)
    print(f"  800 dreams vs 50 dreams: w/l/d {w}/{l}/{d}")

    print("\nBut dreams are slow, and random dreams are blind: MCTS wastes")
    print("its budget imagining futures any club player would dismiss at a")
    print("glance. What it lacks is INTUITION — a brain that whispers 'these")
    print("two moves are the only ones worth dreaming about.'")
    print("You happen to have built one of those in Module 7.")
