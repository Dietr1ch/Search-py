"""
Definitions for a Search Space and Search Problems over them.
"""

import copy
from enum import Enum
from typing import Iterable, List, Sequence, Tuple


class Space:
    """A generic search space."""

    class State:
        """A state in the Search Space."""

        def __hash__(self):
            """The hash of this state."""
            raise NotImplementedError("")

        def __str__(self) -> str:
            """The string representation of this state."""
            raise NotImplementedError("")

        def __eq__(self, other: object) -> bool:
            """Compares 2 states."""
            if not isinstance(other, Space.State):
                return NotImplemented
            raise NotImplementedError(
                "The '{}' State does not implement __eq__ yet".format(self.__class__)
            )

    class Action(Enum):
        """A generic action."""

        # pylint: disable=unused-argument
        # pylint: disable=no-self-use
        def cost(self, state):
            """The cost of executing this action."""
            return 0

        def __str__(self) -> str:
            """The string representation of this action."""
            return str(self.value)

    def neighbors(self, state: State) -> Iterable[Tuple[Action, State]]:
        """The possible actions and their resulting State."""
        raise NotImplementedError("")

    def execute(self, state: State, action: Action) -> State:
        """Applies an action into some State.

        Reuses the neighbors(state) as this is not performance critical.
        """
        # pylint: disable=invalid-name
        for a, s in self.neighbors(state):
            if a == action:
                return copy.deepcopy(s)

        # Something is wrong, let's try to explain the current state.
        action_strs = [str(a) for a, _ in self.neighbors(state)]
        raise ValueError(
            "Received an action that can't be performed at this State. Can't perform {} from {}. Can only perform {}".format(
                action, state, ", ".join(action_strs)
            )
        )

    def to_str(self, problem, state: State) -> str:
        """Formats a Space to a string. Not a problem over it."""
        raise NotImplementedError(
            "The '{}' Space does not implement to_str yet".format(self.__class__)
        )

    @classmethod
    def heuristics(cls, problem):
        """Returns a sorted list of heuristic functions for a given problem.

        The heuristics will be sorted in decreasing quality (and likely cost).
        """
        return []


class RandomAccessSpace(Space):
    """A generic Search Space where random States can be generated."""

    def random_state(self) -> Space.State:
        """Gets a random State with a Uniform distribution."""
        raise NotImplementedError("")


class Problem:
    """A generic problem definition that uses a goal function."""

    def __init__(self, space: Space, starting_states: Sequence[Space.State]):
        self.space = space
        self.starting_states = starting_states

    def is_goal(self, state: Space.State) -> bool:
        """Checks if a state is a goal for this Problem."""
        raise NotImplementedError("")

    def to_str(self, state: Space.State) -> str:
        """The string representing some state over this Problem."""
        return self.space.to_str(problem=self, state=state)

    def start_to_str(self) -> str:
        """The string representing the starting states of this Problem."""
        if len(self.starting_states) == 0:
            raise RuntimeError("This problem does not have starting states.")

        if len(self.starting_states) == 1:
            unique_starting_state = next(iter(self.starting_states))
            return self.to_str(unique_starting_state)

        problem_str = "There's {} starting states,\n".format(len(self.starting_states))
        for starting_state in self.starting_states:
            problem_str += self.to_str(starting_state)
            problem_str += "\n"
        return problem_str

    def all_heuristics(self):
        """Returns a sorted list of heuristic functions for a given problem.

        The heuristics will be sorted in decreasing quality (and likely cost).
        """
        return []


class Heuristic:
    """A heuristic function for a Problem.

    This allows having many functions for a single problem.
    """

    # pylint: disable=invalid-name,no-self-use,unused-argument
    def __init__(self, problem):
        """Creates the Heuristic function for a specific problem.

        Making this an object allows precomputing instance-specific values.
        """
        self.problem = problem

    def __str__(self) -> str:
        """The name of this heuristic."""
        return self.__class__.__name__

    def __call__(self, state: Space.State):
        """The estimated cost of reaching the goal."""
        return 0
