"""Module 9 solution — Imagination."""

import math
import random


class Minimax:
    """Perfect play through exhaustive imagination.

    No learning. No memory of past games. It simply imagines EVERY
    possible future and picks the move that leads to the best guaranteed
    outcome. On tic-tac-toe this is godlike. On Connect Four it would
    need longer than the age of the universe. That gap is this module's
    whole story.
    """

    def __init__(self):
        self._cache = {}

    def select_move(self, game, state):
        _, move = self._search(game, state)
        return move

    def _search(self, game, state):
        if state in self._cache:
            return self._cache[state]
        winner = game.winner(state)
        if winner is not None:
            return (float(winner), None)
        player = game.current_player(state)
        best_value, best_move = None, None
        for move in game.legal_moves(state):
            value, _ = self._search(game, game.next_state(state, move))
            if best_value is None or value * player > best_value * player:
                best_value, best_move = value, move
        self._cache[state] = (best_value, best_move)
        return best_value, best_move


class MCTS:
    """Imagination on a budget.

    Where minimax must imagine everything, MCTS imagines a fixed number
    of plausible futures — spending more of its budget on moves that are
    looking good (exploitation) while still occasionally trying doubtful
    ones (exploration). The UCB rule below is the casino lesson from
    Module 2, reborn inside a tree.
    """

    def __init__(self, n_simulations=400, c_explore=1.4, seed=None):
        self.n_simulations = n_simulations
        self.c_explore = c_explore
        self._rng = random.Random(seed)

    def select_move(self, game, state):
        visits = {}
        totals = {}
        for _ in range(self.n_simulations):
            self._simulate(game, state, visits, totals)
        moves = game.legal_moves(state)
        return max(moves, key=lambda m: visits.get((state, m), 0))

    def _simulate(self, game, state, visits, totals):
        path = []
        # 1. SELECT: descend the known tree, balancing win-rate vs curiosity.
        while game.winner(state) is None:
            moves = game.legal_moves(state)
            untried = [m for m in moves if (state, m) not in visits]
            if untried:
                # 2. EXPAND: step once into the unknown.
                move = self._rng.choice(untried)
                path.append((state, move))
                state = game.next_state(state, move)
                break
            total_visits = sum(visits[(state, m)] for m in moves)
            log_n = math.log(total_visits)
            move = max(moves, key=lambda m: self._ucb(state, m, visits, totals, log_n))
            path.append((state, move))
            state = game.next_state(state, move)
        # 3. ROLLOUT: daydream randomly to the end of the game.
        result = self._rollout(game, state)
        # 4. BACKPROPAGATE: every move on the path learns from the dream.
        for s, m in path:
            mover = game.current_player(s)
            visits[(s, m)] = visits.get((s, m), 0) + 1
            totals[(s, m)] = totals.get((s, m), 0.0) + result * mover

    def _ucb(self, state, move, visits, totals, log_n):
        n = visits[(state, move)]
        average = totals[(state, move)] / n
        curiosity = self.c_explore * math.sqrt(log_n / n)
        return average + curiosity

    def _rollout(self, game, state):
        while True:
            winner = game.winner(state)
            if winner is not None:
                return float(winner)
            move = self._rng.choice(game.legal_moves(state))
            state = game.next_state(state, move)


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

    print("MCTS on Connect Four, imagining only 400 futures per move:")
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
