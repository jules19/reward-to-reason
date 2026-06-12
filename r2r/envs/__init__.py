"""Environments for Reward2Reason.

Every environment speaks the same minimal language:

    state = env.reset()
    state, reward, done = env.step(action)

That's it. Reward is the only teaching signal any agent in this course
will ever receive.
"""

from r2r.envs.bandit import CoinFlip, Casino
from r2r.envs.gridworld import Maze, Cliff, OpenWorld, SMALL_MAZE, BIG_MAZE
from r2r.envs.games import Game, TicTacToe, ConnectFour

__all__ = [
    "CoinFlip",
    "Casino",
    "Maze",
    "Cliff",
    "OpenWorld",
    "SMALL_MAZE",
    "BIG_MAZE",
    "Game",
    "TicTacToe",
    "ConnectFour",
]
