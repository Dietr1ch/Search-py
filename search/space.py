"""
Definitions for a Search Space and Search Problems over them.
"""

from typing import Hashable, Iterable, Set, Tuple


class Space():
    """A generic search space."""

    class Action():
        """A generic action."""

        def cost(self):  # pylint: disable=no-self-use
            """The cost of executing this action."""
            return 0

        def __str__(self) -> str:
            """The string representation of this action."""
            raise NotImplementedError("")

    def neighbors(self, state: Hashable) -> Iterable[Tuple[Action, Hashable]]:
        """The possible actions and their resulting State."""
        raise NotImplementedError("")


class Problem():
    """A generic problem definition that uses a goal function."""

    def __init__(self, space: Space, starts: Set[Hashable]):
        self.space = space

        self.starts = starts

    def is_goal(self, state: Hashable) -> bool:
        """Checks if a state is a goal for this Problem."""
        raise NotImplementedError("")


class SimpleProblem(Problem):
    """A simple problem implementation that has a Set of goal states."""

    def __init__(self,
                 space: Space,
                 starts: Set[Hashable],
                 goal_positions: Set[Hashable]):
        super().__init__(space, starts)

        self.goals = goal_positions

    def is_goal(self, state: Hashable):
        return state in self.goals


class PredefinedSpace(Space):
    """A search space with predefined start and goal states.

    Allows specifying problems for a given Space.
    """

    def starting_states(self) -> Iterable[Hashable]:
        """Generates starting states."""
        raise NotImplementedError("")

    def goal_states(self) -> Iterable[Hashable]:
        """Generates goal states."""
        raise NotImplementedError("")

    def simple_given(self) -> Iterable[SimpleProblem]:
        """Generates problems with a single start and goal."""
        for start in self.starting_states():
            for goal in self.goal_states():
                yield SimpleProblem(self, set([start]), set([goal]))

    def multi_goal_given(self) -> Iterable[SimpleProblem]:
        """Generates problems with a single start and multiple goals."""
        for start in self.starting_states():
            yield SimpleProblem(self, set([start]), set(self.goal_states()))

    def multi_start_and_goal_given(self) -> SimpleProblem:
        """Generates problems with a multiple starts and goals."""
        return SimpleProblem(self,
                             set(self.starting_states()),
                             set(self.goal_states()))


class RandomAccessSpace(Space):
    """A generic search space."""

    def random_state(self) -> Hashable:
        """Gets a random State with a Uniform distribution."""
        raise NotImplementedError("")

    def simple_random(self) -> SimpleProblem:
        """Creates a random problem with a single start and goal."""
        start = self.random_state()
        goal = self.random_state()
        return SimpleProblem(self, set([start]), set([goal]))
