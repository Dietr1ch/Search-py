"""
Definitions for a finite 2D grid space.
"""

# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

import random
from enum import Enum
from typing import Hashable, Iterable, List, Tuple

import numpy as np
from search.space import PredefinedSpace, RandomAccessSpace, Space


class Cell(Enum):
    """Cell contents."""
    EMPTY = " "
    START = "S"
    GOAL = "G"
    WALL = "#"


class Board2D(PredefinedSpace, RandomAccessSpace):
    """A 2D Board with walls with a single Agent.

    Internally the grid represented by a 2d boolean matrix.
    States are represented by cell coordinates of the Agent.
    """

    class Action(Space.Action, Enum):
        """Grid actions."""
        UP = "↑"
        DOWN = "↓"
        LEFT = "←"
        RIGHT = "→"

        def cost(self):  # pylint: disable=no-self-use
            """The cost of executing this action."""
            return 1

        def __str__(self) -> str:
            """The string representation of this action."""
            return str(self.value)

    def __init__(self, grid: List[str]):
        # Store the predefined start and goal states
        self.starts = []
        self.goals = []
        # Store the empty cells to simplify `random_state`
        self.empty_cells = set()

        # Store a the map
        # pylint: disable=invalid-name
        self.H = len(grid)
        self.W = max([len(row) for row in grid])
        self.grid = np.full((self.H, self.W), False)
        for y, row in enumerate(grid):
            for x, char in enumerate(row):
                if char == "#":
                    self.grid[y][x] = True
                    continue

                self.empty_cells.add((x, y))

                if char == "S":
                    self.starts.append((x, y))
                elif char == "G":
                    self.goals.append((x, y))

    def is_wall(self, cell):
        """Checks if a cell has a wall."""
        # pylint: disable=invalid-name
        x, y = cell

        return self.grid[y][x]

    def adjacent_coordinates(
            self, cell: Tuple[int, int]) -> Iterable[Tuple[Board2D.Action, Hashable]]:
        """Generates the actions and resulting cells."""
        # pylint: disable=invalid-name
        x, y = cell

        if x - 1 >= 0:
            yield (Board2D.Action.LEFT, (x - 1, y))
        if x + 1 < self.W:
            yield (Board2D.Action.RIGHT, (x + 1, y))
        if y + 1 < self.H:
            yield (Board2D.Action.DOWN, (x, y + 1))
        if y - 1 >= 0:
            yield (Board2D.Action.UP, (x, y - 1))

    # Space
    # -----
    def neighbors(
            self, state: Hashable) -> Iterable[Tuple[Board2D.Action, Hashable]]:
        """Generates the Actions and neighbor States."""
        # TODO(ddaroch): Figure out a nice way of adding a base class for State

        # pylint: disable=invalid-name
        for a, s in self.adjacent_coordinates(cell=state):
            if not self.is_wall(s):
                yield (a, s)

    # PredefinedSpace
    # ---------------
    def starting_states(self) -> Iterable[Hashable]:
        """The predefined starting states."""
        return self.starts

    def goal_states(self) -> Iterable[Hashable]:
        """The predefined goal states."""
        return self.goals

    # RandomAccessSpace
    # -----------------
    def random_state(self) -> Hashable:
        """Returns a random state."""
        return random.choice(list(self.empty_cells))
