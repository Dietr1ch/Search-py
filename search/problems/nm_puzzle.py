"""
A generalization of the 8-puzzle, the NM-puzzle

An implementation detail that could be surprising, is that the empty space uses
the highest number instead of 0.
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


class NMPuzzle(RandomAccessSpace):
    """A 2D Grid with walls with a single Agent.

    Internally the grid represented by a 2d boolean matrix.
    States are represented by cell coordinates of the Agent.
    """

    class State(Space.State):
        """The game state.

        We store the grid and keep track of the max number to compute neighbors
        faster.

        Members:
          Problem state.
          - grid: The state represented by this node.
          - max_x: The x coordinate of the largest number
          - max_y: The y coordinate of the largest number
          - positions: Map from numbers to their coordinates in the grid.
                       We need this for computing distances quickly, so it's
                       better to compute it just once for every state and cache
                       it's value here once computed
        """

        def __init__(self, grid: np.ndarray, max_x=-1, max_y=-1):
            self.grid = grid
            assert type(self.grid) == np.ndarray

            if max_x == -1 or max_y == -1:
                max_num = 0
                # pylint: disable=invalid-name
                for y, row in enumerate(self.grid):
                    for x, num in enumerate(row):
                        if num > max_num:
                            max_num = num
                            max_x = x
                            max_y = y
            self.max_x: int = max_x
            self.max_y: int = max_y
            assert (0, 0) <= (max_y, max_x) < self.grid.shape

        def __hash__(self) -> int:
            """The hash of this state."""
            # pylint: disable=invalid-name
            salt = 7
            (h, w) = self.grid.shape
            hash_value = hash((h, w))
            for y in range(h):
                for x in range(w):
                    hash_value ^= int(self.grid[y][x] * salt)
                    salt += 1
            return hash_value

        def __str__(self) -> str:
            """The string representation of this state."""
            return "NMPuzzle.State[grid={}]".format(self.grid.tolist())

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, NMPuzzle.State):
                return NotImplemented
            return np.array_equal(self.grid, other.grid)

        def __le__(self, other):
            return self.grid < other.grid

    class Action(Space.Action, Enum):
        """Grid actions."""

        UP = "↑"
        DOWN = "↓"
        LEFT = "←"
        RIGHT = "→"

        def cost(self, state: NMPuzzle.State):  # pylint: disable=no-self-use
            """The cost of executing this action."""
            return 1

    def __init__(self, grid):
        """Creates the Space from a 2D NDArray.

        Note that the puzzle is stored as the state. The space just remembers
        its shape so it's easier to do bound-checks.
        """
        # pylint: disable=invalid-name
        (self.H, self.W) = grid.shape
        self.max_num = self.H * self.W - 1

    def adjacent_coordinates(
        self, cell: Tuple[int, int]
    ) -> Iterable[Tuple[NMPuzzle.Action, Tuple[int, int]]]:
        """Generates the actions and resulting cells."""
        # pylint: disable=invalid-name
        x, y = cell

        # The names of the actions are reversed as we really thing about moving
        # the tile (any number) to the hole (max number).
        if x - 1 >= 0:
            yield (NMPuzzle.Action.RIGHT, (x - 1, y))
        if x + 1 < self.W:
            yield (NMPuzzle.Action.LEFT, (x + 1, y))
        if y + 1 < self.H:
            yield (NMPuzzle.Action.UP, (x, y + 1))
        if y - 1 >= 0:
            yield (NMPuzzle.Action.DOWN, (x, y - 1))

    # Space
    # -----
    def neighbors(
        self, state: Space.State
    ) -> Iterable[Tuple[NMPuzzle.Action, NMPuzzle.State]]:
        """Generates the Actions and neighbor States."""
        if not isinstance(state, NMPuzzle.State):
            raise TypeError("Only NMPuzzle.State is supported")

        # pylint: disable=invalid-name
        hole_position = (state.max_x, state.max_y)
        for a, tile_position in self.adjacent_coordinates(hole_position):
            new_grid = copy.deepcopy(state.grid)
            # Move the neighbor to the hole
            new_grid[hole_position[1]][hole_position[0]] = state.grid[tile_position[1]][
                tile_position[0]
            ]
            # Move the hole to the neighbor
            new_grid[tile_position[1]][tile_position[0]] = self.max_num

            yield (
                a,
                NMPuzzle.State(
                    new_grid, max_x=tile_position[0], max_y=tile_position[1]
                ),
            )

    def to_str(self, problem: Problem, state: Space.State) -> str:
        """Formats a Problem over a NMPuzzle to a colored string.

        Drawing a grid is not enough as we want the problem's start and goal
        states.
        We don't have a nice way to write a generic function printing a generic
        problem based on a generic space printer.
        """
        assert isinstance(problem.space, NMPuzzle)
        assert isinstance(state, NMPuzzle.State)

        space = problem.space
        grid_str = ""
        grid_str += colored(
            ("    █" + ("█" * (5 * space.W)) + "█\n"), "green", attrs=["bold"]
        )

        for y, row in enumerate(state.grid):
            grid_str += colored("%3d " % y, "white")
            grid_str += colored("█", "green", attrs=["bold"])
            for x, num in enumerate(row):
                num_str = "|{:3}|".format(num)
                if num == self.max_num:
                    num_str = "|   |"

                color = "white"
                attrs = []
                if num == self.H * y + x:
                    color = "green"
                    attrs = ["bold"]
                grid_str += colored(num_str, color, attrs=attrs)
            grid_str += colored("█", "green", attrs=["bold"]) + "\n"

        grid_str += colored(
            ("    █" + ("█" * (5 * space.W)) + "█\n"), "green", attrs=["bold"]
        )

        return grid_str

    # RandomAccessSpace
    # -----------------
    def random_state(self) -> NMPuzzle.State:
        """Returns a random state."""
        numbers = list(range(self.H * self.W))
        random.shuffle(numbers)

        grid = np.full((self.H, self.W), 0)
        # pylint: disable=invalid-name
        max_num = 0
        max_x = 0
        max_y = 0
        for y in range(self.H):
            for x in range(self.W):
                grid[y][x] = numbers.pop()
                if grid[y][x] > max_num:
                    max_num = grid[y][x]
                    max_x = x
                    max_y = y

        return NMPuzzle.State(grid, max_x, max_y)


class NMPuzzleProblem(Problem):
    """A simple implementation with a set of goal states."""

    def __init__(
        self,
        space: NMPuzzle,
        starting_states: Sequence[NMPuzzle.State],
        goal_states: Sequence[NMPuzzle.State],
    ):
        super().__init__(space, starting_states)
        self.goal_states = set(goal_states)

    def is_goal(self, state: Space.State) -> bool:
        """Checks if a state is a goal for this Problem."""
        if not isinstance(state, NMPuzzle.State):
            raise TypeError("Only NMPuzzle.State is supported")
        return state in self.goal_states

    @staticmethod
    def all_heuristics():
        """Returns a sorted list of heuristic classes for a given problem.

        The heuristics will be sorted in decreasing quality (and likely cost).
        """
        return [
            NMPuzzleManhattanDistance,
            Heuristic,
        ]


INFINITY = float("inf")


def manhattan_distance_2d(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """The Manhattan distance between a pair of 2D-points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_number_coordinates(state: NMPuzzle.State) -> List[Tuple[int, int]]:
    """Returns a map from tile number to its coordinates.

    Turns a linear lookup in the grid into a constant-time lookup.
    """
    # pylint: disable=invalid-name
    (h, w) = state.grid.shape
    num_map = [(-1, -1) for _ in range(h * w)]

    for y, row in enumerate(state.grid):
        for x, num in enumerate(row):
            num_map[num] = (x, y)

    return num_map


class NMPuzzleManhattanDistance(Heuristic):
    """The Manhattan distance."""

    def __init__(self, problem):
        super().__init__(problem)
        self.goal_number_coordinates = [
            get_number_coordinates(g) for g in self.problem.goal_states
        ]

    def __call__(self, state: Space.State):
        """The estimated cost of reaching the goal."""
        if not isinstance(state, NMPuzzle.State):
            raise TypeError("Only NMPuzzle.State is supported")
        assert self.problem.goal_states, "Some goal states must be defined."

        # TODO: Implement the NMPuzzle Manhattan Distance heuristic.
        return 0


def build_goal_state(height: int, width: int) -> NMPuzzle.State:
    """Builds the canonical Goal State."""
    goal_grid = np.full((height, width), 0)
    tile = 0
    # pylint: disable=invalid-name
    for y in range(height):
        for x in range(width):
            goal_grid[y][x] = tile
            tile += 1

    for y in range(height - 1):
        for x in range(width - 1):
            assert 0 <= goal_grid[y][x] < goal_grid[y + 1][x]
            assert 0 <= goal_grid[y][x] < goal_grid[y][x + 1]

    return NMPuzzle.State(goal_grid)


class NMPuzzleMetaProblem:
    """A Definition of multiple Problems on a single Space.

    A Problem factory focused on a single Space.
    """

    def __init__(self, grid: np.ndarray):
        self.space = NMPuzzle(grid)
        (self.H, self.W) = grid.shape
        # Store the predefined start and its goal state
        self.starting_states: List[NMPuzzle.State] = [NMPuzzle.State(grid)]

        goal_state = build_goal_state(self.H, self.W)
        self.goal_states: Set[NMPuzzle.State] = set([goal_state])

    def simple_given(self):
        """Generates problems with a single start and goal."""
        for starting_state in self.starting_states:
            for goal_state in self.goal_states:
                yield NMPuzzleProblem(
                    self.space, set([starting_state]), set([goal_state])
                )

    def multi_goal_given(self):
        """Generates problems with a single start and all goals.

        This supports generating a single problem, sorting the tiles.
        """
        return iter([])

    def multi_start_and_goal_given(self):
        """Generates problems with a all starts and goals.

        This supports generating a single problem, sorting the tiles.
        """
        return iter([])

    def simple_random(self):
        """Creates a random problem with a single start and goal."""

        # Generate the goal with a random walk on the neighbors to ensure
        # reachability. IIRC 50% of the random states pairs are unreachable as
        # the state graph has 2 huge components
        # This works well because actions are reversible, otherwise we would
        # need to somehow compute the reverse actions.
        goal_state = build_goal_state(self.H, self.W)
        starting_state = copy.deepcopy(goal_state)
        for _ in range(20):
            (_, starting_state) = random.choice(
                list(self.space.neighbors(starting_state))
            )

        starting_states = set([starting_state])
        goal_states = set([goal_state])

        return NMPuzzleProblem(self.space, starting_states, goal_states)
