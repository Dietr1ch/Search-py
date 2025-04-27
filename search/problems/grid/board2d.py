"""
Definitions for a finite 2D grid space.
"""

# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

import copy
import random
from enum import Enum
from typing import Iterable, List, Set, Tuple

import numpy as np
from search.space import Heuristic, Problem, RandomAccessSpace, Space, ZeroHeuristic
from termcolor import colored

INFINITY = float("inf")


def manhattan_distance_2d(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Cell(Enum):
    """Cell contents."""

    EMPTY = " "
    START = "S"
    GOAL = "G"
    WALL = "#"


class Grid2D(RandomAccessSpace):
    """A 2D Grid with walls with a single Agent.

    Internally the grid represented by a 2d boolean matrix.
    States are represented by cell coordinates of the Agent.
    """

    class State(Space.State):
        """The game state."""

        def __init__(self, agent_position: Tuple[int, int]):
            self.agent_position = agent_position

        def __hash__(self):
            """The hash of this state."""
            return hash(self.agent_position)

        def __str__(self) -> str:
            """The string representation of this state."""
            return "Grid2d.State[{}]".format(self.agent_position)

        def __eq__(self, other: Grid2D.State) -> bool:
            """Compares 2 states."""
            return self.agent_position == other.agent_position

    class Action(Space.Action, Enum):
        """Grid actions."""

        UP = "↑"
        DOWN = "↓"
        LEFT = "←"
        RIGHT = "→"

        def cost(self, state: Grid2D.State):  # pylint: disable=no-self-use
            """The cost of executing this action."""
            return 1

    def __init__(self, grid):
        """Creates a grid from a 2D NDArray."""
        self.grid = grid
        (self.H, self.W) = self.grid.shape

        # Store the empty cells to simplify `random_state`
        self.empty_cells = set()
        for y, row in enumerate(grid):
            for x, is_wall in enumerate(row):
                if not is_wall:
                    self.empty_cells.add((x, y))
        # Getting random empty cells uses a list.
        self.empty_cell_list = list(self.empty_cells)

    def is_wall(self, cell):
        """Checks if a cell has a wall."""
        # pylint: disable=invalid-name
        x, y = cell

        return self.grid[y][x]

    def adjacent_coordinates(
        self, cell: Tuple[int, int]
    ) -> Iterable[Tuple[Grid2D.Action, Tuple[int, int]]]:
        """Generates the actions and resulting cells."""
        # pylint: disable=invalid-name
        x, y = cell

        if x - 1 >= 0:
            yield (Grid2D.Action.LEFT, (x - 1, y))
        if x + 1 < self.W:
            yield (Grid2D.Action.RIGHT, (x + 1, y))
        if y + 1 < self.H:
            yield (Grid2D.Action.DOWN, (x, y + 1))
        if y - 1 >= 0:
            yield (Grid2D.Action.UP, (x, y - 1))

    # Space
    # -----
    def neighbors(
        self, state: Grid2D.State
    ) -> Iterable[Tuple[Grid2D.Action, Grid2D.State]]:
        """Generates the Actions and neighbor States."""
        # pylint: disable=invalid-name
        for a, cell in self.adjacent_coordinates(cell=state.agent_position):
            if not self.is_wall(cell):
                yield (a, Grid2D.State(cell))

    def to_str(self, problem: Problem, state: Space.State) -> str:
        """Formats a Problem over a Grid2D to a colored string.

        Drawing a grid is not enough as we want the problem's start and goal
        states.
        We don't have a nice way to write a generic function printing a generic
        problem based on a generic space printer.
        """
        assert isinstance(problem.space, Grid2D)
        assert isinstance(state, Grid2D.State)

        space = problem.space
        grid_str = ""
        grid_str += colored(
            ("    █" + ("█" * (space.W)) + "█\n"), "green", attrs=["bold"]
        )

        for y in range(space.H):
            grid_str += colored("%3d " % y, "white")
            grid_str += colored("█", "green", attrs=["bold"])
            for x in range(space.W):
                char_state = copy.deepcopy(state)
                char_state.agent_position = (x, y)
                if problem.is_goal(char_state):
                    grid_str += colored("G", "yellow", attrs=["bold"])
                elif char_state in problem.starting_states:
                    grid_str += colored("S", "white", attrs=["bold"])
                elif space.is_wall(char_state.agent_position):
                    grid_str += colored("█", "green", attrs=["bold"])
                else:
                    grid_str += " "
            grid_str += colored("█", "green", attrs=["bold"]) + "\n"

        grid_str += colored(
            ("    █" + ("█" * (space.W)) + "█\n"), "green", attrs=["bold"]
        )

        return grid_str
        pass

    # RandomAccessSpace
    # -----------------
    def random_state(self) -> Grid2D.State:
        """Returns a random state."""
        return Grid2D.State(random.choice(self.empty_cell_list))


class Grid2DProblem(Problem):
    """A simple implementation with a set of goal states."""

    def __init__(
        self,
        space: Grid2D,
        starting_states: Set[Grid2D.State],
        goals: Set[Tuple[int, int]],
    ):
        super().__init__(space, starting_states)
        self.goals = goals

    def is_goal(self, state: Grid2D.State) -> bool:
        """Checks if a state is a goal for this Problem."""
        return state.agent_position in self.goals

    def all_heuristics(self) -> List[Heuristic]:
        """Returns a sorted list of heuristic functions for a given problem.

        The heuristics will be sorted in decreasing quality (and likely cost).
        """
        return [
            Grid2DManhattanDistance(self),
            Grid2DSingleDimensionDistance(self),
            Grid2DDiscreteMetric(self),
            ZeroHeuristic(self),
        ]


class Grid2DDiscreteMetric(Heuristic):
    """The Discrete metric, either 0 or 1."""

    def __init__(self, problem):
        super().__init__(problem)

    def __call__(self, state: Grid2D.State):
        """The estimated cost of reaching the goal."""
        if state.agent_position in self.problem.goals:
            return 0
        return 1


class Grid2DSingleDimensionDistance(Heuristic):
    """The Manhattan distance."""

    def __init__(self, problem):
        super().__init__(problem)

    def __call__(self, state: Grid2D.State):
        """The estimated cost of reaching the goal."""
        if self.problem.goals:
            pos = state.agent_position
            return max(
                min([abs(pos[0] - g[0]) for g in self.problem.goals]),
                min([abs(pos[1] - g[1]) for g in self.problem.goals]),
            )
        return INFINITY


class Grid2DManhattanDistance(Heuristic):
    """The Manhattan distance."""

    def __init__(self, problem):
        super().__init__(problem)

    def __call__(self, state: Grid2D.State):
        """The estimated cost of reaching the goal."""
        if self.problem.goals:
            pos = state.agent_position
            return min([manhattan_distance_2d(pos, g) for g in self.problem.goals])
        return INFINITY


class Grid2DMetaProblem:
    """A Definition of multiple Problems on a single Space.

    A Problem factory focused on a single Space.
    """

    def __init__(self, grid_lines: List[str]):
        # Store the predefined start and goal states
        self.starts: List[Grid2D.State] = []
        self.goals: List[Tuple[int, int]] = []

        # Store a the map
        # pylint: disable=invalid-name
        self.H = len(grid_lines)
        self.W = max([len(row) for row in grid_lines])
        grid = np.full((self.H, self.W), False)
        for y, row in enumerate(grid_lines):
            for x, char in enumerate(row):
                if char == "#":
                    grid[y][x] = True
                    continue

                position = (x, y)
                if char == "S":
                    self.starts.append(Grid2D.State(position))
                elif char == "G":
                    self.goals.append(position)

        self.space = Grid2D(grid)

    def simple_given(self):
        """Generates problems with a single start and goal."""
        for start in self.starts:
            for goal in self.goals:
                yield Grid2DProblem(self.space, set([start]), set([goal]))

    def multi_goal_given(self):
        """Generates problems with a single start and all goals."""
        goals = set(self.goals)
        for start in self.starts:
            yield Grid2DProblem(self.space, set([start]), goals)

    def multi_start_and_goal_given(self):
        """Generates problems with a all starts and goals."""
        return Grid2DProblem(self.space, set(self.starts), set(self.goals))

    def simple_random(self):
        """Creates a random problem with a single start and goal."""
        starting_states = set([self.space.random_state()])
        goal_positions = set([self.space.random_state().agent_position])

        return Grid2DProblem(self.space, starting_states, goal_positions)
