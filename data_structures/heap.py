"""
Wrapper for heapq
"""
import heapq
from typing import Any, List, Tuple

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


class Heap:
    """A (min) heap."""

    def __init__(self):
        self.heap: List[Any] = []

    def __bool__(self) -> bool:
        return bool(self.heap)

    def __len__(self) -> int:
        return len(self.heap)

    def __getitem__(self, index: int) -> Any:
        return self.heap[index]

    def slow_find(self, existing_value: Any) -> Tuple[int, Any]:
        for (index, (key, value)) in enumerate(self.heap):
            if value == existing_value:
                return (index, (key, value))
        raise ValueError("'{}' is not in this Heap".format(existing_value))

    def peek(self):
        return self.heap[0]

    def peek_key(self):
        (key, _) = self.heap[0]
        return key

    def peek_value(self):
        (_, value) = self.heap[0]
        return value

    def push(self, key, value):
        heapq.heappush(self.heap, (key, value))

    def pop(self):
        return heapq.heappop(self.heap)

    def is_empty(self) -> bool:
        return len(self.heap) == 0

    def clear(self):
        self.heap = []

    def improve(self, new_key, value_index):
        (old_key, value) = self.heap[value_index]
        assert new_key < old_key

        self.heap[value_index] = (new_key, value)

        # We shouldn't do this, but we have tests to detect breakages and are
        # willing to maintain this on our own.
        # pylint: disable=protected-access
        heapq._siftdown(self.heap, new_key, value_index)
