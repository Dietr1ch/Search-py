"""
A grid with BOMBS!
"""

# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

import copy
import random
from enum import Enum
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np  # type: ignore
from search.space import Heuristic, Problem, RandomAccessSpace, Space
from termcolor import colored

INFINITY = float("inf")


def manhattan_distance_2d(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def hash_sortable_set(sortable_set):
    sorted_set = list(sortable_set)
    sorted_set.sort()
    set_hash = 0
    for item in sorted_set:
        set_hash ^= hash(item)
    return set_hash


class Bombs2D(RandomAccessSpace):
    """A 2D Grid with walls with a single Agent.

    Internally the grid represented by a 2d boolean matrix.
    States are represented by cell coordinates of the Agent.
    """

    class State(Space.State):
        """The game state."""

        def __init__(
            self,
            agent_position: Tuple[int, int],
            bombs: int = 0,
            bomb_positions: Set[Tuple[int, int]] = set(),
        ):
            self.agent_position = agent_position
            self.bombs = bombs
            self.bomb_positions = bomb_positions

            self.destroyed_walls: Set[Tuple[int, int]] = set()

        def __hash__(self):
            """The hash of this state."""
            h = self.bombs
            h ^= hash(self.agent_position)
            h ^= hash_sortable_set(self.agent_position)
            h ^= hash_sortable_set(self.destroyed_walls)
            return h

        def __str__(self) -> str:
            """The string representation of this state."""
            return "Bombs2d.State[pos={} bombs={}({}) destroyed={}]".format(
                self.agent_position,
                self.bombs,
                self.bomb_positions,
                self.destroyed_walls,
            )

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, Bombs2D.State):
                return NotImplemented

            return (
                self.agent_position == other.agent_position
                and self.bombs == other.bombs
                and self.bomb_positions == other.bomb_positions
                and self.destroyed_walls == other.destroyed_walls
            )

    class Action(Space.Action, Enum):
        """Actions on the Bombs2d problem."""

        UP = "↑"
        DOWN = "↓"
        LEFT = "←"
        RIGHT = "→"
        BOMB_UP = "▲"
        BOMB_DOWN = "▼"
        BOMB_LEFT = "◄"
        BOMB_RIGHT = "►"

        def cost(self, state: Bombs2D.State):  # pylint: disable=no-self-use
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

    def is_wall(self, cell: Tuple[int, int], state: State):
        """Checks if a cell has a wall."""
        # pylint: disable=invalid-name
        x, y = cell

        return self.grid[y][x] and cell not in state.destroyed_walls

    def is_bomb(self, cell: Tuple[int, int], state: State):
        """Checks if a position holds a bomb."""
        # pylint: disable=invalid-name,no-self-use
        return cell in state.bomb_positions

    def adjacent_coordinates(
        self, cell: Tuple[int, int]
    ) -> Iterable[Tuple[Bombs2D.Action, Tuple[int, int]]]:
        """Generates the actions and resulting cells."""
        # pylint: disable=invalid-name
        x, y = cell

        if x - 1 >= 0:
            yield (Bombs2D.Action.LEFT, (x - 1, y))
        if x + 1 < self.W:
            yield (Bombs2D.Action.RIGHT, (x + 1, y))
        if y + 1 < self.H:
            yield (Bombs2D.Action.DOWN, (x, y + 1))
        if y - 1 >= 0:
            yield (Bombs2D.Action.UP, (x, y - 1))

    # Space
    # -----
    # pylint: disable=too-many-branches,too-many-statements
    def neighbors(
        self, state: Space.State
    ) -> Iterable[Tuple[Bombs2D.Action, Bombs2D.State]]:
        """Generates the Actions and neighbor States."""
        if not isinstance(state, Bombs2D.State):
            return
        # pylint: disable=invalid-name,too-many-branches,too-many-statements
        x, y = state.agent_position

        # LEFT
        if x - 1 >= 0:
            new_pos = (x - 1, y)

            if self.is_wall(new_pos, state):
                if state.bombs > 0:
                    # BOMB_LEFT
                    new_state = copy.deepcopy(state)
                    new_state.bombs -= 1
                    new_state.destroyed_walls.add(new_pos)
                    yield (Bombs2D.Action.BOMB_LEFT, new_state)
            else:
                # LEFT
                new_state = copy.deepcopy(state)
                if new_pos in state.bomb_positions:
                    new_state.bomb_positions.remove(new_pos)
                    new_state.bombs += 1
                new_state.agent_position = new_pos
                yield (Bombs2D.Action.LEFT, new_state)

        # RIGHT
        if x + 1 < self.W:
            new_pos = (x + 1, y)

            if self.is_wall(new_pos, state):
                if state.bombs > 0:
                    # BOMB_RIGHT
                    new_state = copy.deepcopy(state)
                    new_state.bombs -= 1
                    new_state.destroyed_walls.add(new_pos)
                    yield (Bombs2D.Action.BOMB_RIGHT, new_state)
            else:
                # RIGHT
                new_state = copy.deepcopy(state)
                if new_pos in state.bomb_positions:
                    new_state.bomb_positions.remove(new_pos)
                    new_state.bombs += 1
                new_state.agent_position = new_pos
                yield (Bombs2D.Action.RIGHT, new_state)

        # DOWN
        if y + 1 < self.H:
            new_pos = (x, y + 1)

            if self.is_wall(new_pos, state):
                if state.bombs > 0:
                    # BOMB_DOWN
                    new_state = copy.deepcopy(state)
                    new_state.bombs -= 1
                    new_state.destroyed_walls.add(new_pos)
                    yield (Bombs2D.Action.BOMB_DOWN, new_state)
            else:
                # DOWN
                new_state = copy.deepcopy(state)
                if new_pos in state.bomb_positions:
                    new_state.bomb_positions.remove(new_pos)
                    new_state.bombs += 1
                new_state.agent_position = new_pos
                yield (Bombs2D.Action.DOWN, new_state)

        # UP
        if y - 1 >= 0:
            new_pos = (x, y - 1)

            if self.is_wall(new_pos, state):
                if state.bombs > 0:
                    # BOMB_UP
                    new_state = copy.deepcopy(state)
                    new_state.bombs -= 1
                    new_state.destroyed_walls.add(new_pos)
                    yield (Bombs2D.Action.BOMB_UP, new_state)
            else:
                # UP
                new_state = copy.deepcopy(state)
                if new_pos in state.bomb_positions:
                    new_state.bomb_positions.remove(new_pos)
                    new_state.bombs += 1
                new_state.agent_position = new_pos
                yield (Bombs2D.Action.UP, new_state)

    def to_str(self, problem: Problem, state: Space.State) -> str:
        """Formats a Problem over a Bombs2D to a colored string.

        Drawing a grid is not enough as we want the problem's start and goal
        states.
        We don't have a nice way to write a generic function printing a generic
        problem based on a generic space printer.
        """
        assert isinstance(problem.space, Bombs2D)
        assert isinstance(state, Bombs2D.State)

        space = problem.space
        grid_str = ""
        grid_str += "Bombs: " + colored(str(state.bombs), "red", attrs=["bold"]) + "\n"
        grid_str += colored(
            ("    █" + ("█" * (space.W)) + "█\n"), "green", attrs=["bold"]
        )

        starting_positions = []
        for starting_state in problem.starting_states:
            assert isinstance(starting_state, Bombs2D.State)
            starting_positions.append(starting_state.agent_position)

        for y in range(space.H):
            grid_str += colored("%3d " % y, "white")
            grid_str += colored("█", "green", attrs=["bold"])
            for x in range(space.W):
                position = (x, y)
                new_state = copy.deepcopy(state)
                new_state.agent_position = position
                if problem.is_goal(new_state):
                    grid_str += colored("G", "yellow", attrs=["bold"])
                elif position in starting_positions:
                    grid_str += colored("S", "white", attrs=["bold"])
                elif space.is_bomb(position, state):
                    grid_str += colored("B", "red", attrs=["bold"])
                elif space.is_wall(position, state):
                    grid_str += colored("█", "green", attrs=["bold"])
                else:
                    grid_str += " "
            grid_str += colored("█", "green", attrs=["bold"]) + "\n"

        grid_str += colored(
            ("    █" + ("█" * (space.W)) + "█\n"), "green", attrs=["bold"]
        )

        return grid_str

    # RandomAccessSpace
    # -----------------
    def random_state(self) -> Bombs2D.State:
        """Returns a random state."""
        return Bombs2D.State(random.choice(self.empty_cell_list))


class Bombs2DProblem(Problem):
    """A simple implementation with a set of goal states."""

    def __init__(
        self,
        space: Bombs2D,
        starting_states: Sequence[Bombs2D.State],
        goals: Set[Tuple[int, int]],
    ):
        super().__init__(space, starting_states)
        self.goals = goals

    def is_goal(self, state: Space.State) -> bool:
        """Checks if a state is a goal for this Problem."""
        if not isinstance(state, Bombs2D.State):
            raise TypeError("Only Bombs2D.State is supported")

        return state.agent_position in self.goals

    @staticmethod
    def all_heuristics():
        """Returns a sorted list of heuristic classes for a given problem.

        The heuristics will be sorted in decreasing quality (and likely cost).
        """
        return [
            Bombs2DManhattanDistance,
            Bombs2DSingleDimensionDistance,
            Bombs2DDiscreteMetric,
            Heuristic,
        ]


class Bombs2DDiscreteMetric(Heuristic):
    """The Discrete metric, either 0 or 1."""

    def __call__(self, state: Space.State):
        """The estimated cost of reaching the goal."""
        if not isinstance(state, Bombs2D.State):
            raise TypeError("Only Bombs2D.State is supported")

        if state.agent_position in self.problem.goals:
            return 0
        return 1

    def __str__(self) -> str:
        """The name of this heuristic."""
        return "Bombs2DDiscreteMetric for {}".format(self.problem)


class Bombs2DSingleDimensionDistance(Heuristic):
    """The Manhattan distance."""

    def __call__(self, state: Space.State):
        """The estimated cost of reaching the goal."""
        if not isinstance(state, Bombs2D.State):
            raise TypeError("Only Bombs2D.State is supported")

        if self.problem.goals:
            pos = state.agent_position
            return max(
                min([abs(pos[0] - g[0]) for g in self.problem.goals]),
                min([abs(pos[1] - g[1]) for g in self.problem.goals]),
            )
        return INFINITY

    def __str__(self) -> str:
        """The name of this heuristic."""
        return "Bombs2DSingleDimensionDistance for {}".format(self.problem)


class Bombs2DManhattanDistance(Heuristic):
    """The Manhattan distance."""

    def __call__(self, state: Space.State):
        """The estimated cost of reaching the goal."""
        if not isinstance(state, Bombs2D.State):
            raise TypeError("Only Bombs2D.State is supported")

        if self.problem.goals:
            pos = state.agent_position
            return min([manhattan_distance_2d(pos, g) for g in self.problem.goals])
        return INFINITY

    def __str__(self) -> str:
        """The name of this heuristic."""
        return "Bombs2DManhattanDistance for {}".format(self.problem)


class Cell(Enum):
    """Cell contents."""

    EMPTY = " "
    START = "S"
    GOAL = "G"
    WALL = "#"
    BOMB = "B"


class Bombs2DMetaProblem:
    """A Definition of multiple Problems on a single Space.

    A Problem factory focused on a single Space.
    """

    def __init__(self, grid_lines: List[str], starting_bombs: int):
        # Store the predefined start and goal states
        start_positions: List[Tuple[int, int]] = []
        self.goal_positions: List[Tuple[int, int]] = []
        self.starting_bombs = starting_bombs
        self.starting_bomb_positions: List[Tuple[int, int]] = []

        # Store a the map
        # pylint: disable=invalid-name
        self.H = len(grid_lines)
        self.W = max([len(row) for row in grid_lines])
        grid = np.full((self.H, self.W), False)
        for y, row in enumerate(grid_lines):
            for x, char in enumerate(row):
                if char == Cell.WALL.value:
                    grid[y][x] = True
                    continue

                position = (x, y)
                if char == Cell.START.value:
                    start_positions.append(position)
                elif char == Cell.BOMB.value:
                    self.starting_bomb_positions.append(position)
                elif char == Cell.GOAL.value:
                    self.goal_positions.append(position)
                else:
                    assert char == Cell.EMPTY.value

        self.space = Bombs2D(grid)
        self.starting_states: List[Bombs2D.State] = []
        for start_position in start_positions:
            self.starting_states.append(
                Bombs2D.State(
                    start_position,
                    bombs=self.starting_bombs,
                    bomb_positions=set(self.starting_bomb_positions),
                )
            )

    def simple_given(self):
        """Generates problems with a single start and goal."""
        for start in self.starting_states:
            for goal in self.goal_positions:
                yield Bombs2DProblem(self.space, set([start]), set([goal]))

    def multi_goal_given(self):
        """Generates problems with a single start and all goals."""
        goals = set(self.goal_positions)
        for start in self.starting_states:
            yield Bombs2DProblem(self.space, set([start]), goals)

    def multi_start_and_goal_given(self):
        """Generates problems with a all starts and goals."""
        return Bombs2DProblem(
            self.space, set(self.starting_states), set(self.goal_positions)
        )

    def simple_random(self):
        """Creates a random problem with a single start and goal."""
        starting_states = set([self.space.random_state()])
        goal_positions = set([self.space.random_state().agent_position])

        return Bombs2DProblem(self.space, starting_states, goal_positions)
