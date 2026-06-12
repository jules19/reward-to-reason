"""The Arena — where agents meet, and the Hall of Mirrors Elo ladder.

From Module 8 onward, progress is no longer "did the tests pass" but
"can today's agent beat yesterday's?". The arena plays matches with
colours alternating, and the ladder turns results into Elo so you can
watch your creation climb past its former selves.
"""

import math

from r2r.envs.games import play_game


def play_match(game, agent_a, agent_b, n_games=100):
    """Play n_games with colours alternating.

    Returns (wins_a, wins_b, draws) from agent_a's point of view.
    """
    wins_a = wins_b = draws = 0
    for i in range(n_games):
        if i % 2 == 0:
            result = play_game(game, agent_a, agent_b)
            result_for_a = result
        else:
            result = play_game(game, agent_b, agent_a)
            result_for_a = -result
        if result_for_a > 0:
            wins_a += 1
        elif result_for_a < 0:
            wins_b += 1
        else:
            draws += 1
    return wins_a, wins_b, draws


def score(game, agent_a, agent_b, n_games=100):
    """Match score for agent_a in [0, 1]: wins + half of draws."""
    wins_a, _, draws = play_match(game, agent_a, agent_b, n_games)
    return (wins_a + 0.5 * draws) / n_games


class Ladder:
    """An Elo ladder for a set of named agents.

    >>> ladder = Ladder(TicTacToe())
    >>> ladder.add("random", RandomAgent(seed=0))
    >>> ladder.add("v1", my_agent)
    >>> ladder.run(games_per_pair=50)
    >>> print(ladder.table())
    """

    def __init__(self, game, k=16, base_elo=1000.0):
        self.game = game
        self.k = k
        self.base_elo = base_elo
        self.agents = {}
        self.elo = {}

    def add(self, name, agent):
        self.agents[name] = agent
        self.elo[name] = self.base_elo

    def run(self, games_per_pair=50, rounds=2):
        names = list(self.agents)
        for _ in range(rounds):
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    s = score(self.game, self.agents[a], self.agents[b], games_per_pair)
                    expected = 1.0 / (1.0 + 10 ** ((self.elo[b] - self.elo[a]) / 400.0))
                    delta = self.k * games_per_pair * (s - expected) / 10.0
                    self.elo[a] += delta
                    self.elo[b] -= delta

    def table(self):
        rows = sorted(self.elo.items(), key=lambda kv: -kv[1])
        width = max(len(name) for name in self.elo)
        lines = [f"{'agent':<{width}}  elo", "-" * (width + 7)]
        for name, elo in rows:
            lines.append(f"{name:<{width}}  {elo:6.0f}")
        return "\n".join(lines)


def winrate_bar(s, width=30):
    """A little ASCII bar for a score in [0, 1]."""
    filled = round(s * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {100 * s:.0f}%"
