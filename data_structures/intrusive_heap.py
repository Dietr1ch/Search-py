"""
An intrusive Heap, which improves lookups from O(n) to O(1), making updates
truly O(log(n)) instead of being dominated by the linear cost of find.
"""
from typing import Iterable, List, Tuple

# A heap is a tree-like structure where every subtree's root has a better score
# than all the other nodes in the subtree.
#
# This is often implemented with an array that's traversed in a non-linear way.
# These are the indices we assign to each node.

#                           0
#              1                         2
#       3            4            5             6
#   7      8      9     10    11     12     13     14
# 15 16  17 18  19 20  21 22 23 24  25
#
# The last level will often be incomplete
#
# You can easily go up, down-left, and down-right from any index with,
#   - Up: (i-1)//2
#   - DL: (2*i) + 1
#   - DR: 2(i+1)


# A sentinel, fake-index number to keep track of Nodes out of the Heap
INDEX_NOT_SET = -17


class IntrusiveHeap:
    """A (min) heap that allows to improve keys in log(n) time.

    Intrusive data-structures work on extended values that hold data-structure
    internals to speed up some operations.

    In this case, we will store the internal index within the Heap array on
    each of the Elements we store. That way, looking up the internal position
    on the heap's array for a given Element takes constant time, because it's
    stored right in that Element.

    We don't allow using the same Element in multiple heaps because we don't
    want nor need the overhead of telling which IntrusiveHeap stored what.

    Members:
      a: The list backing this heap.
    """

    class Node:
        """A Node in the Intrusive Heap."""

        def __init__(self):
            self._heap_index = INDEX_NOT_SET

        def __lt__(self, other) -> bool:
            raise NotImplementedError("IntrusiveHeap Nodes must implement __lt__")

    def __init__(self):
        self.a: List[IntrusiveHeap.Node] = []

    def __bool__(self) -> bool:
        return bool(self.a)

    def __len__(self) -> int:
        return len(self.a)

    def __getitem__(self, index: int) -> Node:
        return self.a[index]

    def __contains__(self, node: Node) -> bool:
        if 0 <= node._heap_index < len(self.a):
            # If the node is actually no this heap, then its index must match
            # its position in the array.
            assert node == self.a[node._heap_index]
            return node == self.a[node._heap_index]
        assert node._heap_index == INDEX_NOT_SET
        return False

    def _slow_find(self, existing_node: Node) -> Tuple[int, Node]:
        """Performs a linear-time find.

        Only used to verify key implementation details."""
        for (index, node) in enumerate(self.a):
            if node == existing_node:
                return (index, node)
        raise ValueError("'{}' is not in this Heap".format(existing_node))

    def push(self, node):
        """Pushes a Node into the Heap."""
        # pylint: disable=protected-access
        assert node._heap_index == INDEX_NOT_SET
        node._heap_index = len(self.a)
        self.a.append(node)
        self.sync_improvement(node)

    def peek(self):
        """Peeks at a node with the best score."""
        return self.a[0]

    def peek_best(self) -> Iterable[Node]:
        """Peeks at the nodes tied with the best score."""
        assert len(self.a) > 0
        best_node = self.a[0]
        for i in range(len(self.a)):
            if self.a[i] > best_node:
                return
            yield self.a[i]

    def pop(self):
        """Gets and removes a node with the best score."""
        node = self.a[0]

        # pylint: disable=protected-access
        assert node._heap_index == 0
        node._heap_index = INDEX_NOT_SET

        self.a[0] = None
        self._sync_removal()

        return node

    def is_empty(self) -> bool:
        """Checks whether this Heap is empty."""
        return len(self.a) == 0

    def clear(self):
        """Clear the Heap and Free all its nodes."""
        # pylint: disable=protected-access
        for node in self.a:
            node._head_index = INDEX_NOT_SET
        self.a = []

    def sync_improvement(self, node: Node):
        """Updates the heap to reflect the improved rank of the Node.

        The heap MUST be updated right after the node improves, otherwise its
        index will go out of sync.

        A ripoff of CPython's heapq._siftdown.
        We re-implement this to get access to the new index after improving a key.

        https://github.com/python/cpython/blob/3.9/Lib/heapq.py
        """
        # pylint: disable=protected-access
        assert (
            0 <= node._heap_index < len(self)
        ), "This node is way out of sync. Index out of bounds.."
        assert node == self.a[node._heap_index], "This node is out of sync."

        # Bubble up the value at pos until it's not better than it's parent.
        pos = node._heap_index
        parent = (pos - 1) >> 1
        while pos > 0 and self.a[parent] > self.a[pos]:
            # Swap the node with it's parent as they are in the wrong order
            (self.a[pos], self.a[parent]) = (self.a[parent], self.a[pos])

            # Update their indices
            self.a[pos]._heap_index = pos
            self.a[parent]._heap_index = parent

            # Continue swapping upwards..
            pos = parent
            parent = (pos - 1) >> 1

    def _sync_removal(self):
        """Updates the heap to remove the hole caused by a Heap::pop()

        The heap MUST be updated right after the node improves, otherwise its
        index will go out of sync.

        A ripoff of CPython's heapq._siftup.
        We re-implement this to get access to the new index after improving a key.

        https://github.com/python/cpython/blob/3.9/Lib/heapq.py
        """
        # pylint: disable=protected-access
        hole_index = 0
        child = 2 * hole_index + 1  # The left child, but reused to track the best child
        end = len(self.a)

        while child < end:
            assert self.a[hole_index] is None
            assert self.a[child] is not None

            # Find the best child
            child_r = child + 1
            if child_r < end and self.a[child_r] < self.a[child]:
                child = child_r
            assert self.a[child] is not None

            # Move child up and update its index
            self.a[hole_index] = self[child]
            self.a[hole_index]._heap_index = hole_index

            # Clear the child node and update the indices
            self.a[child] = None
            hole_index = child  # The hole is now here
            child = 2 * hole_index + 1  # And this is the new left child

        # We are done, but now there's a None in our last level.
        # We can keep our array compact by and fill this hole with the last
        # element of the array. It's not safe to just swap, we also need to try
        # to bubble it up after swapping
        assert self.a[hole_index] is None
        if hole_index == end - 1:
            # We got lucky
            self.a.pop()
            return
        self.a[hole_index] = self.a.pop()
        self.a[hole_index]._heap_index = hole_index
        self.sync_improvement(self.a[hole_index])
