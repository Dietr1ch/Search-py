"""
Dijkstra's specialization of a generic search algorithm.
"""

# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

from typing import Dict, Optional

from data_structures.intrusive_heap import IntrusiveHeap

from search.algorithms.search import Node, SearchAlgorithm
from search.space import Space


class Dijkstra(SearchAlgorithm):
    """Best-first Search.

    Implements Open with an intrusive Heap.
    Extends the base search Nodes to store internals.
    """

    class DijkstraNode(Node, IntrusiveHeap.Node):
        """A search Node for running Dijkstra's Search Algorithm

        Members:
          Problem state.
          - state: The state represented by this node.

          Problem costs
          - g: The cost to reach this node

          Path recovery. Could be omitted, but knowing there's a solution
          without being to tell which one is often useless.
          - action: The action used to reach this node
          - parent: The node used to reach this node
        """

        def __init__(
            self,
            state: Space.State,
            action: Optional[Space.Action],
            parent: Optional[Node],
            g,
        ):
            Node.__init__(self, state, action, parent)
            IntrusiveHeap.Node.__init__(self)

            # pylint: disable=invalid-name
            assert g >= 0
            self.g = g

        # search.Node
        # -----------
        def __str__(self) -> str:
            """The string representation of this Node."""
            parent_action = "ø"
            parent_state = "ø"
            if self.parent:
                parent_action = str(self.action)
                parent_state = str(self.parent.state)
            return "DijkstraNode[s={}, a={}, p.s={}, g={}]".format(
                self.state, parent_action, parent_state, self.g
            )

        # IntrusiveHeap.Node
        # ------------------
        def __lt__(self, other) -> bool:
            """Compares the cost of reaching the nodes."""
            return self.g < other.g

    class Open(SearchAlgorithm.Open):
        """An Open set implementation using an intrusive Heap."""

        def __init__(self):
            self.heap: IntrusiveHeap = IntrusiveHeap()
            self.node_map: Dict[Space.State, Dijkstra.DijkstraNode] = dict()

        def insert(self, node: Node):
            """Appends a Node into the Open list."""
            if not isinstance(node, Dijkstra.DijkstraNode):
                raise TypeError("Only AStarNode is supported")

            self.node_map[node.state] = node
            self.heap.push(node)

        def pop(self) -> Dijkstra.DijkstraNode:
            """Takes the first (oldest) Node from the Open list."""
            node = self.heap.pop()
            return self.node_map.pop(node.state)

        def peek(self) -> Node:
            """Looks at the top Node from Open without removing it."""
            return self.heap.peek()

        def __len__(self) -> int:
            """Counts the Nodes in Open."""
            return len(self.heap)

        def __bool__(self) -> bool:
            """Checks if there's Nodes in Open."""
            return len(self.heap) > 0

        def __contains__(self, state: Space.State) -> bool:
            """Checks if there's a Node for a state in Open."""
            return state in self.node_map

        def __getitem__(self, state: Space.State) -> Node:
            return self.node_map[state]

        def sync_improvement(self, node: Dijkstra.DijkstraNode):
            """Updates the internal heap to keep up with a node improvement.

            It must be called right after any node gets a better score.
            """
            self.heap.sync_improvement(node)

    @classmethod
    def name(cls) -> str:
        """Returns the name of the Algorithm."""
        return "Dijkstra's Algorithm"

    @classmethod
    def create_open(cls) -> Open:
        """Returns the container to use for the Open set."""
        return Dijkstra.Open()

    def create_starting_node(self, state: Space.State) -> Node:
        """Create an Starting Node."""
        self.nodes_created += 1
        return Dijkstra.DijkstraNode(state, action=None, parent=None, g=0)

    def reach(self, state: Space.State, action: Space.Action, parent: Node):
        """Reaches a state and updates Open."""
        if not isinstance(parent, Dijkstra.DijkstraNode):
            raise TypeError("Only DijkstraNode is supported")

        # pylint: disable=invalid-name
        g = parent.g + action.cost(parent.state)

        if state not in self.open:
            self.nodes_created += 1
            self.open.insert(
                Dijkstra.DijkstraNode(state, action=action, parent=parent, g=g)
            )
            return

        # The state already had a node in Open, but maybe we found a better way
        # to reach it, if so, we would like to update the node.
        old_node = self.open[state]
        if g >= old_node.g:
            # Meh..
            return

        # We found a better path!
        self.nodes_updated += 1
        old_node.action = action
        old_node.parent = parent
        old_node.g = g
        self.open.sync_improvement(old_node)
