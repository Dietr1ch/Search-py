"""
Definitions for a finite 2D grid space.
"""

# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

import copy
import random
from enum import Enum
from typing import Iterable, List, Tuple

import numpy as np
from search.space import PredefinedSpace, Problem, RandomAccessSpace, Space
from termcolor import colored


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

    class State(Space.State):
        def __init__(self, x: int, y: int):
            self.position = (x, y)

        def __hash__(self):
            """The hash of this state."""
            return hash(self.position)

        def __str__(self) -> str:
            """The string representation of this state."""
            return "Board2d.State[{}]".format(self.position)

        def __eq__(self, other: Board2D.State) -> bool:
            """Compares 2 states."""
            return self.position == other.position

    class Action(Space.Action, Enum):
        """Grid actions."""
        UP = "↑"
        DOWN = "↓"
        LEFT = "←"
        RIGHT = "→"

        def cost(self, state: Board2D.State):  # pylint: disable=no-self-use
            """The cost of executing this action."""
            return 1

        def __str__(self) -> str:
            """The string representation of this action."""
            return str(self.value)

    def __init__(self, grid: List[str]):
        # Store the predefined start and goal states
        self.starts: List[Board2D.State] = []
        self.goals: List[Board2D.State] = []
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
                    self.starts.append(Board2D.State(x, y))
                elif char == "G":
                    self.goals.append(Board2D.State(x, y))

    def is_wall(self, cell):
        """Checks if a cell has a wall."""
        # pylint: disable=invalid-name
        x, y = cell

        return self.grid[y][x]

    def adjacent_coordinates(
            self, cell: Tuple[int, int]) -> Iterable[Tuple[Board2D.Action, Tuple[int, int]]]:
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
            self, state: Board2D.State) -> Iterable[Tuple[Board2D.Action, Board2D.State]]:
        """Generates the Actions and neighbor States."""
        # pylint: disable=invalid-name
        for a, cell in self.adjacent_coordinates(cell=state.position):
            if not self.is_wall(cell):
                new_state: Board2D.State = copy.deepcopy(state)
                new_state.position = cell
                yield (a, new_state)

    def to_str(self, problem: Problem, state: State) -> str:
        """Formats a Problem over a Board2D to a colored string.

        Drawing a board is not enough as we want the problem's start and goal
        states.
        We don't have a nice way to write a generic function printing a generic
        problem based on a generic space printer.
        """
        board_str = ''
        board_str += colored(('    █' + ('█' * (self.W)) +
                             '█\n'), 'green', attrs=['bold'])

        for y in range(self.H):
            board_str += colored("%3d " % y, 'white')
            board_str += colored("█", 'green', attrs=['bold'])
            for x in range(self.W):
                char_state = copy.deepcopy(state)
                char_state.position = (x, y)
                if problem.is_goal(char_state):
                    board_str += colored('G', 'yellow', attrs=['bold'])
                elif char_state in problem.starts:
                    board_str += colored('S', 'white', attrs=['bold'])
                elif self.is_wall(char_state.position):
                    board_str += colored('█', 'green', attrs=['bold'])
                else:
                    board_str += " "
            board_str += colored('█', 'green', attrs=['bold']) + '\n'

        board_str += colored(('    █' + ('█' * (self.W)) +
                             '█\n'), 'green', attrs=['bold'])

        return board_str

    # PredefinedSpace
    # ---------------
    def starting_states(self) -> Iterable[Board2D.State]:
        """The predefined starting states."""
        return self.starts

    def goal_states(self) -> Iterable[Board2D.State]:
        """The predefined goal states."""
        return self.goals

    # RandomAccessSpace
    # -----------------
    def random_state(self) -> Board2D.State:
        """Returns a random state."""
        r_x, r_y = random.choice(list(self.empty_cells))
        return Board2D.State(r_x, r_y)
