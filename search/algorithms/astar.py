"""
AStar's specialization of a generic search algorithm.
"""

from typing import Dict, List, Optional

from data_structures.intrusive_heap import IntrusiveHeap

from search.algorithms.search import Node, HeuristicSearchAlgorithm
from search.space import Space


class AStar(HeuristicSearchAlgorithm):
    """Breadth-first Search.

    Implements Open with a List and a set.
    It uses the base Node class as we don't need to extend it.
    """

    class AStarNode(Node, IntrusiveHeap.Node):
        """A search Node for running A* Search Algorithm

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
            h,
        ):
            Node.__init__(self, state, action, parent)
            IntrusiveHeap.Node.__init__(self)

            # pylint: disable=invalid-name
            assert g >= 0
            assert h >= 0
            self.g = g
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
            return "AStarNode[s={}, a={}, p.s={}, g={}, h={}]".format(
                self.state, parent_action, parent_state, self.g, self.h
            )

        # IntrusiveHeap.Node
        # ------------------
        def __lt__(self, other) -> bool:
            """Returns < of "(f, h)" to perform informed/optimistic tie-breaking."""
            return (self.g + self.h, self.h) < (other.g + other.h, other.h)

    class Open(HeuristicSearchAlgorithm.Open):
        """An Open set implementation using a Queue."""

        def __init__(self):
            self.heap: IntrusiveHeap = IntrusiveHeap()
            self.node_map: Dict[Space.State, Node] = dict()

        def insert(self, node: Node):
            """Appends a Node into the Open list."""
            self.node_map[node.state] = node
            self.heap.push(node)

        def pop(self) -> Node:
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

        def sync_improvement(self, node):
            """Updates the internal heap to keep up with a node improvement.

            It must be called right after any node gets a better score.
            """
            self.heap.sync_improvement(node)

    @classmethod
    def name(cls) -> str:
        """Returns the name of the Algorithm."""
        return "A* Algorithm"

    @classmethod
    def create_open(cls) -> Open:
        """Returns the container to use for the Open set."""
        return AStar.Open()

    def create_starting_node(self, state: Space.State) -> Node:
        """Create an Starting Node."""
        self.nodes_created += 1
        return AStar.AStarNode(state, action=None, parent=None, g=0, h=self.h(state))

    def reach(self, state: Space.State, action: Space.Action, parent: AStarNode):
        """Reaches a state and updates Open."""
        # pylint: disable=invalid-name
        g = parent.g + action.cost(parent.state)

        if state not in self.open:
            self.nodes_created += 1
            self.open.insert(
                AStar.AStarNode(state, action=action, parent=parent, g=g, h=self.h(state))
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
