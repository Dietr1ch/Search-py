"""
Tests for Dijkstra's Search algorithm.
"""

from typing import Optional

from search.algorithms.dijkstra import Dijkstra
from search.algorithms.search import Node, SearchAlgorithm
from search.problems.grid.board2d import Grid2DMetaProblem
from search.space import Problem


def test_no_solution():
    metaproblem = Grid2DMetaProblem(
        [
            "     ",
            " ####",
            "     ",
            "#### ",
            "     ",
            "S    ",
        ]
    )
    problem: Problem = next(iter(metaproblem.multi_goal_given()))
    dijkstra: SearchAlgorithm = Dijkstra(problem)

    # Search
    goal_node: Optional[Node] = dijkstra.search()

    # A solution must be found
    assert goal_node is None
    # This maps needs to be completely expanded
    assert dijkstra.expansions == 22
    assert 10_000 < dijkstra.time_ns < 10_000_000


def test_expansion_order():
    length = 100
    metaproblems = [
        Grid2DMetaProblem(
            [
                "G" + " " * length + "S" + " " * length,
            ]
        ),
        Grid2DMetaProblem(
            [
                " " * length + "S" + " " * length + "G",
            ]
        ),
    ]

    # pylint: disable=invalid-name
    for mp in metaproblems:
        problem: Problem = next(iter(mp.multi_goal_given()))
        dijkstra: SearchAlgorithm = Dijkstra(problem)

        # Search
        goal_node: Optional[Node] = dijkstra.search()

        assert goal_node is not None
        assert goal_node.path(problem.space) is not None
        assert 2 * length < dijkstra.expansions <= 2 * (length + 1)
        assert 100_000 < dijkstra.time_ns < 100_000_000
        assert dijkstra.states_reached > 2 * length
        assert dijkstra.states_generated > 4 * length  # Expansions generate ~2 states
        assert dijkstra.nodes_created > 2 * length
        assert dijkstra.nodes_updated == 0  # The graph becomes a "tree" with Closed :/
