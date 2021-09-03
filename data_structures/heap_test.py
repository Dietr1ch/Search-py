import random

from data_structures.heap import Heap


def test_rank():
    # pylint: disable=invalid-name
    a = object()
    b = object()
    c = object()
    assert a != b

    heap = Heap()
    assert len(heap) == 0
    # Add 1:a and 2:b 3:c
    heap.push(2, b)
    heap.push(1, a)
    heap.push(3, c)
    assert len(heap) == 3

    # Retrieve them in order
    assert (1, a) == heap.pop()
    assert len(heap) == 2
    assert (2, b) == heap.pop()
    assert len(heap) == 1
    assert (3, c) == heap.pop()
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
    a = object()
    b = object()
    assert a != b

    heap = Heap()
    heap.push(1, a)
    heap.push(2, b)
    assert heap.peek() == (1, a)

    (b_i, _) = heap.slow_find(b)
    assert b_i == 1
    heap.improve(0, b_i)

    assert heap.peek() == (0, b)
    assert (0, (0, b)) == heap.slow_find(b)
    assert (1, (1, a)) == heap.slow_find(a)

    assert (0, b) == heap.pop()
    assert (1, a) == heap.pop()
    assert len(heap) == 0


def test_ordered_updates():
    random.seed(1)

    numbers = [(i * i % 7) for i in range(100)]

    heap = Heap()
    for num in numbers:
        heap.push(num, str(num))

    best_key = float("-inf")
    while heap:
        (key, _) = heap.pop()
        assert key >= best_key

        best_key = key

    assert len(heap) == 0
