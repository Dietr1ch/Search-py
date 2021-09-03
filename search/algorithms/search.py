"""
Definitions for a generic Search Algorithm.
"""

# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

import copy
import time
from typing import Iterable, List, Optional, Tuple

from termcolor import colored

from search.space import Space


class Node:
    """A search Node. Maintains the problem state and the path that reached it.

    Members:
      Problem state.
      - state: The state represented by this node.

      Path recovery. Could be omitted, but knowing there's a solution without
      being to tell which one is often useless.
      - action: The action used to reach this node
      - parent: The node used to reach this node

    Note that the cost can be computed as the sum of the costs in the path,
    however, good cost-aware implementations will extend Node to cache it.
    """

    def __init__(
        self, state: Space.State, action: Optional[Space.Action], parent: Optional[Node]
    ):
        self.state: Space.State = state

        self.action = action
        self.parent = parent

    def path(self, space: Space) -> Path:
        """Gets the initial state and actions used to reach this node."""
        return Path(space, self)

    def __str__(self) -> str:
        """The string representation of this Node."""
        parent_action = "ø"
        parent_state = "ø"
        if self.parent:
            parent_action = str(self.action)
            parent_state = str(self.parent.state)
        return "Node[s={}, a={}, p.s={}]".format(
            self.state, parent_action, parent_state
        )


class Path:
    """A path in the Search Tree.

    Computes the full path on initialization to help free the Nodes.
    """

    def __init__(self, space: Space, goal_node: Node):
        self.space: Space = space
        self.end: Space.State = copy.deepcopy(goal_node.state)
        self.path: List[Space.Action] = []

        node = goal_node
        while node.parent is not None:
            assert node.parent is not None
            assert node.action is not None

            self.path.append(node.action)
            node = node.parent
        self.path.reverse()

        assert node.parent is None
        assert node.action is None

        self.start = copy.deepcopy(node.state)

        # Precompute cost
        current_state = copy.deepcopy(self.start)
        self.path_cost = 0
        for action in self.path:
            self.path_cost += action.cost(current_state)
            current_state = space.execute(current_state, action)
        assert current_state == self.end

    def starting_state(self) -> Space.State:
        """The State where this Path begins."""
        return self.start

    def final_state(self) -> Space.State:
        """The State at the end of this Path."""
        return self.end

    def actions(self) -> Iterable[Space.Action]:
        """The full sequence of actions to reach a state."""
        return self.path

    def __len__(self) -> int:
        """The length of this Path."""
        return len(self.path)

    def cost(self) -> int:
        """The cost of this Path.

        This is recomputed from the State and Actions. It discards information
        from the search to avoid trusting every implementation.
        """
        return self.path_cost

    def compressed_actions(self) -> Iterable[Tuple[int, Space.Action]]:
        """A run-length encoding of the Path."""
        if not self.path:
            return

        last_action = self.path[0]
        count = 0

        for action in self.path:
            if action == last_action:
                count += 1
                continue
            yield (count, last_action)
            count = 1
            last_action = action
        yield (count, last_action)

    def __str__(self) -> str:
        return "Path[c:{}, l:{}, p:{} => {} => {}]".format(
            self.cost(),
            len(self),
            colored(str(self.starting_state()), "green", attrs=["bold"]),
            colored(
                " ".join([str(n) + str(a) for n, a in self.compressed_actions()]),
                "blue",
                attrs=["bold"],
            ),
            colored(str(self.final_state()), "yellow", attrs=["bold"]),
        )


class SearchAlgorithm:
    """A generic search algorithm.

    Members:
        problem: The problem being solved.

        open: Nodes that might lead to a solution.
        closed: States that were already explored.
    """

    def __init__(self, problem):
        self.problem = problem

        self.open = self.create_open()
        self.closed = set()

        # Statistics
        self.expansions: int = 0
        self.time_ns: Optional[int] = None

    class Open:
        """A generic Open set.

        Ideally it should support:
        - Quick insertion of Nodes (O(1))
        - Quick removal of a best Node (O(1))
        - Allow implementing reach efficiently. Generally:
          - Quick random Node lookup to avoid duplicating nodes in Open (O(1))
          - Fast Node update in case better paths are found (log(n))
        """

        def insert(self, node: Node):
            """Inserts a Node into Open."""
            raise NotImplementedError("")

        def pop(self) -> Node:
            """Takes a Node from Open."""
            raise NotImplementedError("")

        def __len__(self) -> int:
            """Counts the Nodes in Open."""
            raise NotImplementedError("")

        def __bool__(self):
            """Checks if there's Nodes in Open."""
            raise NotImplementedError("")

    @classmethod
    def name(cls) -> str:
        """Returns the name of the Algorithm."""
        raise NotImplementedError("")

    @classmethod
    def create_open(cls) -> Open:
        """Returns the container to use for the Open set."""
        raise NotImplementedError("")

    @classmethod
    def create_starting_node(cls, state: Space.State) -> Node:
        """Creates an Starting Node."""
        raise NotImplementedError("")

    def reach(self, state: Space.State, action: Space.Action, parent: Node):
        """Reaches a state and updates Open"""
        raise NotImplementedError("")

    def _actually_search(self) -> Optional[Node]:
        """Finds a single goal Node."""
        for start in self.problem.starting_states:
            self.open.insert(self.create_starting_node(start))

        while self.open:
            node = self.open.pop()

            if self.problem.is_goal(node.state):
                return node

            # Expand the node and consider all its neighboring states.
            self.expansions += 1
            self.closed.add(node.state)
            for action, state in self.problem.space.neighbors(node.state):
                if state in self.closed:
                    # Déjà vu, we reached an expanded state.
                    continue  # Not falling for this (again?).

                self.reach(state, action, parent=node)

        return None

    def search(self) -> Optional[Node]:
        """Finds a single goal Node."""
        self.time_ns = time.perf_counter_ns()
        solution = self._actually_search()
        self.time_ns = time.perf_counter_ns() - self.time_ns

        return solution
