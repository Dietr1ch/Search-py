"""
Tests for the Breadth-first Search algorithm.
"""

from typing import Optional

from search.algorithms.bfs import BFS
from search.algorithms.search import Node, SearchAlgorithm
from search.problems.grid.board2d import Grid2D, Grid2DMetaProblem
from search.space import Problem, Space


def test_no_solution():
    metaproblem = Grid2DMetaProblem([
        "     ",
        " ####",
        "     ",
        "#### ",
        "     ",
        "S    ",
    ])
    problem: Problem = next(iter(metaproblem.multi_goal_given()))
    bfs: SearchAlgorithm = BFS(problem)

    # Search
    goal_node: Optional[Node] = bfs.search()

    # A solution must be found
    assert goal_node is None
    # This maps needs to be completely expanded
    assert bfs.expansions == 22
    assert 10_000 < bfs.time_ns < 10_000_000


def test_expansion_order():
    length = 100
    metaproblem = Grid2DMetaProblem([
        "G" + " " * length + "S" + " " * length,
    ])
    problem: Problem = next(iter(metaproblem.multi_goal_given()))
    bfs: SearchAlgorithm = BFS(problem)

    # Search
    goal_node: Optional[Node] = bfs.search()

    assert goal_node is not None
    assert goal_node.path(problem.space) is not None
    assert 2 * length < bfs.expansions <= 2 * (length + 1)
    assert 100_000 < bfs.time_ns < 100_000_000
