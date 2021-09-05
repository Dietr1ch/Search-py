import random

from data_structures.intrusive_heap import IntrusiveHeap


class TestNode(IntrusiveHeap.Node):
    def __init__(self, key):
        IntrusiveHeap.Node.__init__(self)
        self.key = key

    def improve_key(self, new_key):
        """Makes the key for a node better."""
        assert new_key < self.key
        self.key = new_key

    # IntrusiveHeap.Node
    # ------------------
    def __lt__(self, other) -> bool:
        return self.key < other.key


def test_rank():
    # pylint: disable=invalid-name
    a = TestNode("2")
    b = TestNode("1")
    c = TestNode("3")
    assert a != b

    heap = IntrusiveHeap()
    assert len(heap) == 0
    heap.push(b)
    heap.push(a)
    heap.push(c)
    assert len(heap) == 3

    # Retrieve them in order
    assert b == heap.pop()
    assert len(heap) == 2
    assert a == heap.pop()
    assert len(heap) == 1
    assert c == heap.pop()
    assert len(heap) == 0

    fails = False
    # pylint: disable=bare-except
    try:
        heap.pop()
    except:
        fails = True

    assert fails


def test_update():
    # pylint: disable=invalid-name
    a = TestNode("1")
    b = TestNode("2")
    assert a != b

    heap = IntrusiveHeap()
    heap.push(a)
    heap.push(b)
    assert a == heap.peek()

    # pylint: protected-access
    (b_index, _) = heap._slow_find(b)
    assert b._heap_index == b_index
    assert b_index == 1
    b.improve_key("0")
    heap.sync_improvement(b)

    assert heap.peek() == b
    assert (0, b) == heap._slow_find(b)
    assert (1, a) == heap._slow_find(a)

    assert b == heap.pop()
    assert a == heap.pop()
    assert len(heap) == 0


def test_sort_nodes():
    random.seed(1)

    nodes = [TestNode((i * i % 7)) for i in range(100)]

    heap = IntrusiveHeap()
    for node in nodes:
        heap.push(node)

    best_key = float("-inf")
    while heap:
        node = heap.pop()
        assert node.key >= best_key

        best_key = node.key

    assert len(heap) == 0
