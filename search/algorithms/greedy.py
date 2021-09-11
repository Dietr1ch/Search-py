"""
A* specialization of a generic search algorithm.
"""

# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

from typing import Dict, Optional

from data_structures.intrusive_heap import IntrusiveHeap

from search.algorithms.search import HeuristicSearchAlgorithm, Node, SearchAlgorithm
from search.space import Heuristic, Problem, Space


class Greedy(HeuristicSearchAlgorithm):
    """Greedy Search Algorithm

    Implements Open with an intrusive Heap.
    Extends the base search Nodes to store internals.
    """

    class GreedyNode(Node, IntrusiveHeap.Node):
        """A search Node for running Greedy Search Algorithm

        Members:
          Problem state.
          - state: The state represented by this node.

          Problem costs
          - h: The estimated cost of reaching a goal from this node.

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
            h,
        ):
            Node.__init__(self, state, action, parent)
            IntrusiveHeap.Node.__init__(self)

            # pylint: disable=invalid-name
            assert h >= 0
            self.h = h

        # search.Node
        # -----------
        def __str__(self) -> str:
            """The string representation of this Node."""
            parent_action = "ø"
            parent_state = "ø"
            if self.parent:
                parent_action = str(self.action)
                parent_state = str(self.parent.state)
            return "GreedyNode[s={}, a={}, p.s={}, h={}]".format(
                self.state, parent_action, parent_state, self.h
            )

        # IntrusiveHeap.Node
        # ------------------
        def __lt__(self, other) -> bool:
            """Sorts the nods based on their h-value."""
            return self.h < other.h

    class Open(SearchAlgorithm.Open):
        """An Open set implementation using an intrusive Heap."""

        def __init__(self):
            self.heap: IntrusiveHeap = IntrusiveHeap()
            self.node_map: Dict[Space.State, Greedy.GreedyNode] = dict()

        def insert(self, node: Node):
            """Appends a Node into the Open list."""
            if not isinstance(node, Greedy.GreedyNode):
                raise TypeError("Only GreedyNode is supported")

            self.node_map[node.state] = node
            self.heap.push(node)

        def pop(self) -> Greedy.GreedyNode:
            """Takes the first (oldest) Node from the Open list."""
            node = self.heap.pop()
            return self.node_map.pop(node.state)

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

        def sync_improvement(self, node: Greedy.GreedyNode):
            """Updates the internal heap to keep up with a node improvement.

            It must be called right after any node gets a better score.
            """
            self.heap.sync_improvement(node)

    def __init__(self, problem: Problem, heuristic: Heuristic):
        super().__init__(problem, heuristic)

    @classmethod
    def name(cls) -> str:
        """Returns the name of the Algorithm."""
        return "Greedy Algorithm"

    @classmethod
    def create_open(cls) -> Open:
        """Returns the container to use for the Open set."""
        return Greedy.Open()

    def create_starting_node(self, state: Space.State) -> Node:
        """Create an Starting Node."""
        self.nodes_created += 1
        return Greedy.GreedyNode(state, action=None, parent=None, h=self.h(state))

    def reach(self, state: Space.State, action: Space.Action, parent: Node):
        """Reaches a state and updates Open."""
        if not isinstance(parent, Greedy.GreedyNode):
            raise TypeError("Only GreedyNode is supported")

        if state in self.open:
            # It's h-value won't improve, there's nothing to update.
            return

        self.nodes_created += 1
        self.open.insert(
            Greedy.GreedyNode(state, action=action, parent=parent, h=self.h(state))
        )
