"""Gridworld environments — the first worlds with *places*.

States are (row, col) tuples. Actions are 0=up, 1=right, 2=down, 3=left.
Reward is +1.0 on reaching the goal (episode ends), 0.0 otherwise.
Episodes also end after `max_steps` to keep lost agents from wandering
forever.
"""

import random

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
ACTION_NAMES = ["up", "right", "down", "left"]

SMALL_MAZE = """\
#########
#S..#...#
#.#.#.#.#
#.#...#.#
#.#####.#
#.....#.#
###.#.#.#
#...#..G#
#########"""

BIG_MAZE = """\
#####################
#S....#.......#.....#
#.###.#.#####.#.###.#
#.#...#.....#.#.#...#
#.#.#######.#.#.#.###
#.#.#.....#.#...#...#
#.#.#.###.#.#######.#
#...#.#.#.#.......#.#
###.#.#.#.#######.#.#
#...#.#.........#.#.#
#.###.#########.#.#.#
#.#...#.......#.#...#
#.#.###.#####.#.###.#
#.#.#...#...#.#...#.#
#.#.#.###.#.#.###.#.#
#.#.#.#...#.#.#...#.#
#.#.#.#.###.#.#.#####
#.#...#.#...#.#.....#
#.#####.#.###.#####.#
#.......#...........G
#####################"""


class Maze:
    """A fixed maze parsed from an ASCII map. Modules 3 and 4.

    `slip` is the probability that the floor is icy and your action is
    replaced by a random one. A slip of 0.0 is a dependable world; Module 4
    turns the ice on.
    """

    n_actions = 4

    def __init__(self, ascii_map=SMALL_MAZE, max_steps=400, slip=0.0, seed=None):
        self.grid = [list(line) for line in ascii_map.strip("\n").splitlines()]
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.max_steps = max_steps
        self.slip = slip
        self._rng = random.Random(seed)
        self.start = self._find("S")
        self.goal = self._find("G")
        self._pos = self.start
        self._steps = 0

    def _find(self, char):
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell == char:
                    return (r, c)
        raise ValueError(f"map has no {char!r} cell")

    def reset(self):
        self._pos = self.start
        self._steps = 0
        return self._pos

    def step(self, action):
        if self.slip > 0.0 and self._rng.random() < self.slip:
            action = self._rng.randrange(4)
        dr, dc = ACTIONS[action]
        r, c = self._pos
        nr, nc = r + dr, c + dc
        if self.grid[nr][nc] != "#":
            self._pos = (nr, nc)
        self._steps += 1
        done = self._pos == self.goal
        reward = 1.0 if done else 0.0
        if self._steps >= self.max_steps:
            done = True
        return self._pos, reward, done

    def render(self, pos=None):
        pos = pos if pos is not None else self._pos
        lines = []
        for r, row in enumerate(self.grid):
            line = []
            for c, cell in enumerate(row):
                line.append("@" if (r, c) == pos else cell)
            lines.append("".join(line))
        return "\n".join(lines)

    def states(self):
        """All non-wall cells. Handy for measuring table coverage."""
        return [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if self.grid[r][c] != "#"
        ]


class Cliff:
    """The cliff walk. Module 4's world.

    A 4x12 ledge. Start at the bottom-left, goal at the bottom-right.
    Between them, along the bottom row, is a cliff. Reaching the goal pays
    +1. Stepping into the cliff pays -1 and ends the episode. The shortest
    safe walk hugs the cliff edge the whole way.

    This world exposes the difference between learning "what happened to
    me" and learning "what the world is".
    """

    n_actions = 4
    n_rows, n_cols = 4, 12

    def __init__(self, max_steps=200):
        self.start = (3, 0)
        self.goal = (3, 11)
        self.max_steps = max_steps
        self._pos = self.start
        self._steps = 0

    def is_cliff(self, pos):
        r, c = pos
        return r == 3 and 0 < c < 11

    def reset(self):
        self._pos = self.start
        self._steps = 0
        return self._pos

    def step(self, action):
        dr, dc = ACTIONS[action]
        r, c = self._pos
        nr = min(max(r + dr, 0), self.n_rows - 1)
        nc = min(max(c + dc, 0), self.n_cols - 1)
        self._pos = (nr, nc)
        self._steps += 1
        if self.is_cliff(self._pos):
            return self._pos, -1.0, True
        if self._pos == self.goal:
            return self._pos, 1.0, True
        done = self._steps >= self.max_steps
        return self._pos, 0.0, done

    def render(self, path=()):
        marks = set(path)
        lines = []
        for r in range(self.n_rows):
            line = []
            for c in range(self.n_cols):
                if (r, c) == self.start:
                    line.append("S")
                elif (r, c) == self.goal:
                    line.append("G")
                elif self.is_cliff((r, c)):
                    line.append("~")
                elif (r, c) in marks:
                    line.append("o")
                else:
                    line.append(".")
            lines.append("".join(line))
        return "\n".join(lines)


class OpenWorld:
    """An open room where the goal MOVES every episode. Modules 5 and 7.

    Every reset places the agent and the goal at fresh random cells, so the
    thing an agent must learn is not "the path" but the *idea* of walking
    toward a goal. A state here is ((agent_r, agent_c), (goal_r, goal_c)).

    On a size-12 board that is 144 x 143 ≈ 20,000 distinct states. A Q-table
    treats every one of them as an unrelated universe. This is the Wall.
    """

    n_actions = 4

    def __init__(self, size=12, max_steps=80, seed=None):
        self.size = size
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self._agent = (0, 0)
        self.goal = (size - 1, size - 1)
        self._steps = 0

    def reset(self):
        s = self.size
        self._agent = (self._rng.randrange(s), self._rng.randrange(s))
        while True:
            self.goal = (self._rng.randrange(s), self._rng.randrange(s))
            if self.goal != self._agent:
                break
        self._steps = 0
        return (self._agent, self.goal)

    def step(self, action):
        dr, dc = ACTIONS[action]
        r, c = self._agent
        nr = min(max(r + dr, 0), self.size - 1)
        nc = min(max(c + dc, 0), self.size - 1)
        self._agent = (nr, nc)
        self._steps += 1
        done = self._agent == self.goal
        reward = 1.0 if done else 0.0
        if self._steps >= self.max_steps:
            done = True
        return (self._agent, self.goal), reward, done

    def n_states(self):
        s2 = self.size * self.size
        return s2 * (s2 - 1)

    def encode(self, state):
        """Turn a state into 4 numbers in [0, 1] — food for a neural network."""
        (ar, ac), (gr, gc) = state
        d = self.size - 1
        return [ar / d, ac / d, gr / d, gc / d]

    def render(self, state=None):
        agent, goal = state if state is not None else (self._agent, self.goal)
        lines = []
        for r in range(self.size):
            line = []
            for c in range(self.size):
                if (r, c) == agent:
                    line.append("@")
                elif (r, c) == goal:
                    line.append("G")
                else:
                    line.append(".")
            lines.append("".join(line))
        return "\n".join(lines)
