"""
Breadth-first Search specialization of a generic search algorithm.
"""

from typing import List, Set

from search.algorithms.search import Node, SearchAlgorithm
from search.space import Space


class BFS(SearchAlgorithm):
    """Breadth-first Search.

    Implements Open with a List and a set.
    It uses the base Node class as we don't need to extend it.
    """

    class Open(SearchAlgorithm.Open):
        """An Open set implementation using a Queue."""

        def __init__(self):
            self.nodes: List[Node] = []
            self.states: Set[Space.State] = set()

        def insert(self, node: Node):
            """Appends a Node into the Open list."""
            self.states.add(node.state)
            self.nodes.append(node)

        def pop(self) -> Node:
            """Takes the first (oldest) Node from the Open list."""
            node = self.nodes.pop(0)
            self.states.remove(node.state)
            return node

        def __len__(self) -> int:
            """Counts the Nodes in Open."""
            return len(self.nodes)

        def __bool__(self) -> bool:
            """Checks if there's Nodes in Open."""
            return len(self.nodes) > 0

        def __contains__(self, state: Space.State) -> bool:
            """Checks if there's a Node for a state in Open."""
            return state in self.states

    @classmethod
    def name(cls) -> str:
        """Returns the name of the Algorithm."""
        return "Breadth-first Search"

    @classmethod
    def create_open(cls) -> Open:
        """Returns the container to use for the Open set."""
        return BFS.Open()

    # pylint: no-self-argument
    def create_starting_node(self, state: Space.State) -> Node:
        """Create an Starting Node."""
        self.nodes_created += 1
        return Node(state, action=None, parent=None)

    def reach(self, state: Space.State, action: Space.Action, parent: Node):
        """Reaches a state and updates Open."""
        if state in self.open:
            # If the state was already in Open, then we know that this new path
            # to it is not better.
            return

        self.nodes_created += 1
        self.open.insert(Node(state, action, parent))
