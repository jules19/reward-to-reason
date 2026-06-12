"""Module 8 — The Mirror.

Take away the teacher. Take away the opponent. Lock the agent in a room
with its own reflection and nothing but the final score of each game.

Read README.md first.
Check your work:   pytest tests/test_module08.py
See it live:       python modules/08-the-mirror/mirror.py
"""

import random


class SelfPlayLearner:
    """An agent that becomes its own teacher.

    values[state] estimates the eventual result of the game from this
    position, from player +1's point of view (+1 means X wins, -1 means
    O wins). The agent plays BOTH sides against itself; after each game,
    every position that occurred is nudged toward the actual result.
    """

    def __init__(self, game, learning_rate=0.1, epsilon=0.2, seed=None):
        self.game = game
        self.values = {}
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self._rng = random.Random(seed)

    def value(self, state):
        """Estimated final result from +1's point of view.

        If the state is already decided (game.winner is not None), return
        the actual winner as a float — terminal positions are facts, not
        opinions. Otherwise look it up in self.values, default 0.0.
        """
        # TODO 1
        raise NotImplementedError("TODO 1")

    def best_move(self, state, rng=None):
        """The move whose resulting position looks best FOR THE MOVER.

        Careful — this is the one subtle point of the module. values are
        from +1's point of view, but the mover might be -1. The mover
        prefers the move m maximizing:

            current_player(state) * value(next_state(state, m))

        Break ties randomly (rng or self._rng).
        """
        # TODO 2
        raise NotImplementedError("TODO 2")

    def select_move(self, game, state):
        """Arena interface: greedy play, no exploration."""
        return self.best_move(state)

    def play_training_game(self):
        """One game against the mirror. Returns the list of ALL states
        seen, from the initial position to the terminal one.

        Both sides are you: at each turn, with probability epsilon play a
        random legal move, otherwise best_move."""
        # TODO 3
        raise NotImplementedError("TODO 3")

    def learn_from_game(self, history):
        """Nudge every NON-terminal position in the game toward the final
        result (Module 1's rule, one more disguise):

            values[s] <- values[s] + lr * (result - values[s])
        """
        # TODO 4
        raise NotImplementedError("TODO 4")

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
