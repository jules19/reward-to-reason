"""Module 8 solution — The Mirror."""

import random


class SelfPlayLearner:
    """An agent that becomes its own teacher.

    values[state] estimates the eventual result of the game from this
    position, from player +1's point of view (+1 means X wins, -1 means
    O wins). The agent plays BOTH sides against itself; after each game,
    every position that occurred is nudged toward the actual result.

    Reward only arrives at the very end of the game, exactly like the
    maze — but here, the agent IS the environment's other half.
    """

    def __init__(self, game, learning_rate=0.1, epsilon=0.2, seed=None):
        self.game = game
        self.values = {}
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self._rng = random.Random(seed)

    def value(self, state):
        """Estimated final result from +1's point of view."""
        w = self.game.winner(state)
        if w is not None:
            return float(w)
        return self.values.get(state, 0.0)

    def best_move(self, state, rng=None):
        """The move whose resulting position looks best FOR THE MOVER."""
        rng = rng or self._rng
        player = self.game.current_player(state)
        moves = self.game.legal_moves(state)
        scored = [(player * self.value(self.game.next_state(state, m)), m)
                  for m in moves]
        best = max(s for s, _ in scored)
        return rng.choice([m for s, m in scored if s == best])

    def select_move(self, game, state):
        """Arena interface: greedy play, no exploration."""
        return self.best_move(state)

    def play_training_game(self):
        """One game against the mirror. Returns the list of states seen."""
        state = self.game.initial_state()
        history = [state]
        while self.game.winner(state) is None:
            if self._rng.random() < self.epsilon:
                move = self._rng.choice(self.game.legal_moves(state))
            else:
                move = self.best_move(state)
            state = self.game.next_state(state, move)
            history.append(state)
        return history

    def learn_from_game(self, history):
        """Nudge every position in the game toward the final result."""
        result = float(self.game.winner(history[-1]))
        for state in history[:-1]:
            old = self.values.get(state, 0.0)
            self.values[state] = old + self.learning_rate * (result - old)

    def train(self, n_games):
        for _ in range(n_games):
            self.learn_from_game(self.play_training_game())


if __name__ == "__main__":
    from r2r.arena import play_match
    from r2r.builtin import MinimaxAgent, RandomAgent
    from r2r.envs import TicTacToe

    game = TicTacToe()
    agent = SelfPlayLearner(game, seed=0)

    print("An agent alone in a room with a mirror. No opponent. No teacher.")
    print("No rules of strategy. Only: the final result of each game.\n")
    for batch in range(6):
        agent.train(5000)
        w, l, d = play_match(game, agent, RandomAgent(seed=1), n_games=200)
        wm, lm, dm = play_match(game, agent, MinimaxAgent(), n_games=2)
        print(f"  after {5000 * (batch + 1):5d} mirror games: "
              f"vs random w/l/d {w}/{l}/{d}   vs PERFECT play w/l/d {wm}/{lm}/{dm}")

    print(f"\n  positions it now has opinions about: {len(agent.values):,}")
    print("\nIt taught itself to never lose — even against a perfect player —")
    print("by playing nobody but its own reflection. Where did the knowledge")
    print("come from? Not from a teacher. From the reward, bouncing between")
    print("the two sides of the mirror.")
