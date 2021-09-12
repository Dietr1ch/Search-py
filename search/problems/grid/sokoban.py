"""
An implementation of the classic Sokoban game.
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


class Sokoban(RandomAccessSpace):
    """A Sokoban Space. A 2D Grid with a single agent, and walls.

    Internally the grid represented by a 2d boolean matrix.
    """

    class State(Space.State):
        """The game state.

        Defines the position of the agent and all the boxes.
        """

        def __init__(
            self,
            agent_position: Tuple[int, int],
            box_positions: Set[Tuple[int, int]],
        ):
            self.agent_position = agent_position
            self.box_positions = box_positions

        def __hash__(self):
            """The hash of this state."""
            # pylint: disable=invalid-name
            h = hash(self.agent_position)
            h ^= hash_sortable_set(self.box_positions)
            return h

        def __str__(self) -> str:
            """The string representation of this state."""
            return "Sokoban.State[pos={} boxes={}]".format(
                self.agent_position,
                self.box_positions,
            )

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, Sokoban.State):
                return NotImplemented

            return (
                self.agent_position == other.agent_position
                and self.box_positions == other.box_positions
            )

        def has_box(self, cell: Tuple[int, int]):
            """Checks if a cell has a box."""
            return cell in self.box_positions

    class Action(Space.Action, Enum):
        """Actions on the Sokoban problem."""

        UP = "↑"
        DOWN = "↓"
        LEFT = "←"
        RIGHT = "→"
        PUSH_UP = "▲"
        PUSH_DOWN = "▼"
        PUSH_LEFT = "◄"
        PUSH_RIGHT = "►"

        def cost(self, state: Sokoban.State):  # pylint: disable=no-self-use
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

    def is_wall(self, cell: Tuple[int, int]):
        """Checks if a cell has a wall."""
        assert self.is_valid(cell)

        # pylint: disable=invalid-name
        x, y = cell
        return self.grid[y][x]

    def is_valid(self, cell: Tuple[int, int]):
        """Checks if a cell is valid."""
        # pylint: disable=invalid-name
        x, y = cell
        return 0 <= x < self.W and 0 <= y < self.H

    def is_clear(self, cell: Tuple[int, int], state: State):
        """Checks if a cell has a box."""
        assert self.is_valid(cell)

        if self.is_wall(cell):
            return False
        if state.has_box(cell):
            return False
        return True

    # Space
    # -----
    # pylint: disable=too-many-arguments
    def _yield_actions(
        self,
        state: Sokoban.State,
        walk_action: Action,
        push_action: Action,
        new_pos: Tuple[int, int],
        push_pos: Tuple[int, int],
    ) -> Iterable[Tuple[Sokoban.Action, Sokoban.State]]:
        if not self.is_valid(new_pos):
            return iter([])

        if self.is_wall(new_pos):
            return iter([])

        # Push
        if state.has_box(new_pos):
            if not self.is_valid(push_pos):
                return iter([])
            if not self.is_clear(push_pos, state):
                return iter([])
            new_state = copy.deepcopy(state)
            # Move agent to where the box was
            new_state.agent_position = new_pos
            # Move box behind
            new_state.box_positions.remove(new_pos)
            assert len(new_state.box_positions) < len(state.box_positions)
            new_state.box_positions.add(push_pos)
            assert len(new_state.box_positions) == len(state.box_positions)
            yield (push_action, new_state)
        else:
            if not self.is_wall(new_pos):
                new_state = copy.deepcopy(state)
                new_state.agent_position = new_pos
                yield (walk_action, new_state)

    # pylint: disable=too-many-branches,too-many-statements
    def neighbors(
        self, state: Space.State
    ) -> Iterable[Tuple[Sokoban.Action, Sokoban.State]]:
        """Generates the Actions and neighbor States."""
        if not isinstance(state, Sokoban.State):
            return
        # pylint: disable=invalid-name
        x, y = state.agent_position

        # LEFT
        if x - 1 >= 0:
            for a, s in self._yield_actions(
                state=state,
                walk_action=Sokoban.Action.LEFT,
                push_action=Sokoban.Action.PUSH_LEFT,
                new_pos=(x - 1, y),
                push_pos=(x - 2, y),
            ):
                yield (a, s)
        # RIGHT
        if x + 1 < self.W:
            for a, s in self._yield_actions(
                state=state,
                walk_action=Sokoban.Action.RIGHT,
                push_action=Sokoban.Action.PUSH_RIGHT,
                new_pos=(x + 1, y),
                push_pos=(x + 2, y),
            ):
                yield (a, s)
        # DOWN
        if y + 1 < self.H:
            for a, s in self._yield_actions(
                state=state,
                walk_action=Sokoban.Action.DOWN,
                push_action=Sokoban.Action.PUSH_DOWN,
                new_pos=(x, y + 1),
                push_pos=(x, y + 2),
            ):
                yield (a, s)
        # UP
        if y - 1 >= 0:
            for a, s in self._yield_actions(
                state=state,
                walk_action=Sokoban.Action.UP,
                push_action=Sokoban.Action.PUSH_UP,
                new_pos=(x, y - 1),
                push_pos=(x, y - 2),
            ):
                yield (a, s)

    def to_str(self, problem: Problem, state: Space.State) -> str:
        """Formats a Sokoban problem to a colored string.

        Drawing a grid is not enough as we want the problem's start and goal
        states.
        We don't have a nice way to write a generic function printing a generic
        problem based on a generic space printer.
        """
        assert isinstance(problem, SokobanProblem)
        assert isinstance(problem.space, Sokoban)
        assert isinstance(state, Sokoban.State)

        grid_str = ""
        grid_str += colored(
            ("    █" + ("█" * (self.W)) + "█\n"), "green", attrs=["bold"]
        )

        starting_positions = []
        for starting_state in problem.starting_states:
            assert isinstance(starting_state, Sokoban.State)
            starting_positions.append(starting_state.agent_position)

        # pylint: disable=invalid-name
        for y in range(self.H):
            grid_str += colored("%3d " % y, "white")
            grid_str += colored("█", "green", attrs=["bold"])
            for x in range(self.W):
                position = (x, y)

                # Walls
                if self.is_wall(position):
                    grid_str += colored("█", "green", attrs=["bold"])
                # Boxes
                elif state.has_box(position):
                    if position in problem.goal_positions:
                        grid_str += colored("B", "green", attrs=["bold"])
                    else:
                        grid_str += colored("B", "red", attrs=["bold"])
                # Agent
                elif position == state.agent_position:
                    grid_str += colored("a", "white", attrs=["bold"])
                elif position in problem.goal_positions:
                    grid_str += colored("G", "yellow", attrs=["bold"])
                # Empty
                else:
                    grid_str += " "
            grid_str += colored("█", "green", attrs=["bold"]) + "\n"

        grid_str += colored(
            ("    █" + ("█" * (self.W)) + "█\n"), "green", attrs=["bold"]
        )

        return grid_str

    # RandomAccessSpace
    # -----------------
    def random_state(self) -> Sokoban.State:
        """Returns a random state."""
        box_positions = set([random.choice(self.empty_cell_list)])
        return Sokoban.State(
            agent_position=random.choice(self.empty_cell_list),
            box_positions=box_positions,
        )


class SokobanProblem(Problem):
    """A simple implementation with a set of goal states."""

    def __init__(
        self,
        space: Sokoban,
        starting_states: Sequence[Sokoban.State],
        goal_positions: Set[Tuple[int, int]],
    ):
        super().__init__(space, starting_states)
        self.goal_positions = goal_positions

    def is_goal(self, state: Space.State) -> bool:
        """Checks if every box is in a goal position for this Problem."""
        if not isinstance(state, Sokoban.State):
            raise TypeError("Only Sokoban.State is supported")

        # pylint: disable=use-a-generator
        return all([(box in self.goal_positions) for box in state.box_positions])

    @staticmethod
    def all_heuristics():
        """Returns a sorted list of heuristic classes for a given problem.

        The heuristics will be sorted in decreasing quality (and likely cost).
        """
        return [
            SokobanBetterDistance,
            SokobanSimpleManhattanDistance,
            SokobanMisplacedBoxes,
            SokobanDiscreteMetric,
            Heuristic,
        ]


class SokobanDiscreteMetric(Heuristic):
    """The Discrete metric, either 0 or 1."""

    def __call__(self, state: Space.State):
        """The estimated cost of reaching the goal."""
        if not isinstance(state, Sokoban.State):
            raise TypeError("Only Sokoban.State is supported")

        if self.problem.is_goal(state):
            return 0
        return 1


class SokobanMisplacedBoxes(Heuristic):
    """An heuristic counting misplaced boxes."""

    def __call__(self, state: Space.State):
        """The estimated cost of reaching the goal."""
        if not isinstance(state, Sokoban.State):
            raise TypeError("Only Sokoban.State is supported")

        if not self.problem.goal_positions:
            return INFINITY

        misplaced_boxes = 0
        for box in state.box_positions:
            if box not in self.problem.goal_positions:
                misplaced_boxes += 1
        return misplaced_boxes


class SokobanSimpleManhattanDistance(Heuristic):
    """The Manhattan distance.

    The distance between each box and the closest goal.
    """

    def __call__(self, state: Space.State):
        """The estimated cost of reaching the goal."""
        if not isinstance(state, Sokoban.State):
            raise TypeError("Only Sokoban.State is supported")

        if not self.problem.goal_positions:
            return INFINITY
        # TODO: Implement the Manhattan distance.
        return 0


class SokobanBetterDistance(Heuristic):
    """A distance better than the Manhattan distance."""

    def __call__(self, state: Space.State):
        """The estimated cost of reaching the goal."""
        if not isinstance(state, Sokoban.State):
            raise TypeError("Only Sokoban.State is supported")

        if not self.problem.goal_positions:
            return INFINITY
        # TODO: Implement an heuristic better than the manhattan distance.
        return 0


class Cell(Enum):
    """Cell contents."""

    EMPTY = " "
    WALL = "#"
    START = "S"
    GOAL = "G"
    BOX = "B"
    BOX_AND_GOAL = "!"


class SokobanMetaProblem:
    """A Definition of multiple Problems on a single Sokoban Space.

    A Problem factory focused on a single Space.
    """

    def __init__(self, grid_lines: List[str]):
        # Store the predefined start and goal states
        start_positions: List[Tuple[int, int]] = []
        self.goal_positions: List[Tuple[int, int]] = []
        self.starting_box_positions: List[Tuple[int, int]] = []

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
                elif char == Cell.BOX.value:
                    self.starting_box_positions.append(position)
                elif char == Cell.GOAL.value:
                    self.goal_positions.append(position)
                elif char == Cell.BOX_AND_GOAL.value:
                    self.starting_box_positions.append(position)
                    self.goal_positions.append(position)
                else:
                    assert char == Cell.EMPTY.value

        self.space = Sokoban(grid)
        self.starting_states: List[Sokoban.State] = []
        for start_position in start_positions:
            self.starting_states.append(
                Sokoban.State(
                    agent_position=start_position,
                    box_positions=set(self.starting_box_positions),
                )
            )

    def simple_given(self):
        """Generates problems with a single start and goal."""
        for start in self.starting_states:
            for goal in self.goal_positions:
                yield SokobanProblem(self.space, set([start]), set([goal]))

    def multi_goal_given(self):
        """Generates problems with a single start and all goals."""
        goal_positions = set(self.goal_positions)
        for start in self.starting_states:
            yield SokobanProblem(self.space, set([start]), goal_positions)

    def multi_start_and_goal_given(self):
        """Generates problems with a all starts and goals."""
        return SokobanProblem(
            self.space, set(self.starting_states), set(self.goal_positions)
        )

    def simple_random(self):
        """Creates a random problem with a single start and goal."""
        starting_states = set([self.space.random_state()])
        goal_positions = set([self.space.random_state().agent_position])

        return SokobanProblem(self.space, starting_states, goal_positions)
