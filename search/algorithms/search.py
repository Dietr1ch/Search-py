"""
Definitions for a generic Search Algorithm.
"""

# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

from typing import Hashable, List, Optional, Tuple

from search.space import Space


class Node():
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

    def __init__(self,
                 state: Hashable,
                 action: Optional[Space.Action],
                 parent: Optional[Node]):
        self.state = state

        self.action = action
        self.parent = parent

    def path(self) -> Tuple[Hashable, List[Space.Action], Hashable]:
        """Gets the initial state and actions used to reach this node."""
        path: List[Space.Action] = []

        node = self
        ending_state = node.state

        while node.parent:
            assert(node.action is not None)
            action: Space.Action = node.action
            path.append(action)
            node = node.parent
        starting_state = node.state

        path.reverse()
        return (starting_state, path, ending_state)


class SearchAlgorithm():
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
        self.expansions = 0

    class Open():
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
    def create_starting_node(cls, state: Hashable) -> Node:
        """Creates an Starting Node."""
        raise NotImplementedError("")

    def reach(self, state: Hashable, action: Space.Action, parent: Node):
        """Reaches a state and updates Open"""
        raise NotImplementedError("")

    def search(self) -> Optional[Node]:
        """Finds a single goal Node."""
        for start in self.problem.starts:
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
