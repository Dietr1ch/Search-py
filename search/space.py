"""
Definitions for a Search Space and Search Problems over them.
"""

import copy
from typing import Iterable, Optional, Set, Tuple


class Space():
    """A generic search space."""

    class State():
        """A state in the Search Space."""
        def __hash__(self):
            """The hash of this state."""
            raise NotImplementedError("")
        def __str__(self) -> str:
            """The string representation of this state."""
            raise NotImplementedError("")
        def __eq__(self, other) -> bool:
            """Compares 2 states."""
            raise NotImplementedError("")

    class Action():
        """A generic action."""

        # pylint: disable=unused-argument
        # pylint: disable=no-self-use
        def cost(self, state):
            """The cost of executing this action."""
            return 0

        def __str__(self) -> str:
            """The string representation of this action."""
            raise NotImplementedError("")

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
            "Received an action that can't be performed at this State. Can't perform {} from {}. Can only perform {}".format(action, state, ", ".join(action_strs)))

    def to_str(self, problem, state: State) -> str:
        """Formats a Problem over a Board2D to an ASCII colored string."""
        raise NotImplementedError("")


class Problem():
    """A generic problem definition that uses a goal function."""

    def __init__(self, space: Space, starts: Set[Space.State]):
        self.space = space

        self.starts = starts

    def is_goal(self, state: Space.State) -> bool:
        """Checks if a state is a goal for this Problem."""
        raise NotImplementedError("")

    def to_str(self, state: Optional[Space.State] = None) -> str:
        """Formats a Problem over to a string.

        Only one of the starting state is depicted.
        """
        if state is None:
            state = next(iter(self.starts))
        return self.space.to_str(self, state)


class SimpleProblem(Problem):
    """A simple problem implementation that has a Set of goal states."""

    def __init__(self,
                 space: Space,
                 starts: Set[Space.State],
                 goal_positions: Set[Space.State]):
        super().__init__(space, starts)

        self.goals = goal_positions

    def is_goal(self, state: Space.State):
        return state in self.goals


class PredefinedSpace(Space):
    """A search space with predefined start and goal states.

    Allows specifying problems for a given Space.
    """

    def starting_states(self) -> Iterable[Space.State]:
        """Generates starting states."""
        raise NotImplementedError("")

    def goal_states(self) -> Iterable[Space.State]:
        """Generates goal states."""
        raise NotImplementedError("")

    def simple_given(self) -> Iterable[SimpleProblem]:
        """Generates problems with a single start and goal."""
        for start in self.starting_states():
            for goal in self.goal_states():
                yield SimpleProblem(self, set([start]), set([goal]))

    def multi_goal_given(self) -> Iterable[SimpleProblem]:
        """Generates problems with a single start and all goals."""
        for start in self.starting_states():
            yield SimpleProblem(self, set([start]), set(self.goal_states()))

    def multi_start_and_goal_given(self) -> SimpleProblem:
        """Generates problems with a all starts and goals."""
        return SimpleProblem(self,
                             set(self.starting_states()),
                             set(self.goal_states()))


class RandomAccessSpace(Space):
    """A generic search space."""

    def random_state(self) -> Space.State:
        """Gets a random State with a Uniform distribution."""
        raise NotImplementedError("")

    def simple_random(self) -> SimpleProblem:
        """Creates a random problem with a single start and goal."""
        start = self.random_state()
        goal = self.random_state()
        return SimpleProblem(self, set([start]), set([goal]))
