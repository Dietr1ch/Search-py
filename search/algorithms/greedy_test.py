"""
Tests for A* Search algorithm.
"""

from typing import Optional

from search.algorithms.greedy import Greedy
from search.algorithms.search import HeuristicSearchAlgorithm, Node
from search.problems.grid.board2d import (
    Grid2D,
    Grid2DManhattanDistance,
    Grid2DMetaProblem,
)
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
    heuristic = Grid2DManhattanDistance(problem)
    greedy: HeuristicSearchAlgorithm = Greedy(problem, heuristic)

    # Search
    goal_node: Optional[Node] = greedy.search()

    # A solution must be found
    assert goal_node is None
    # This maps needs to be completely expanded
    assert greedy.expansions == 22
    assert 10_000 < greedy.time_ns < 10_000_000


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
        heuristic = Grid2DManhattanDistance(problem)
        greedy: HeuristicSearchAlgorithm = Greedy(problem, heuristic)

        # Search
        goal_node: Optional[Node] = greedy.search()

        assert goal_node is not None
        assert goal_node.path(problem.space) is not None
        assert length < greedy.expansions <= (length + 1)
        assert 100_000 < greedy.time_ns < 100_000_000
        assert length < greedy.states_reached <= length + 2
        assert (
            2 * length <= greedy.states_generated <= 2 * (length + 1)
        )  # Expansions generate ~2 states
        assert greedy.nodes_created == length + 3  # wrong way + s + length + goal
        assert greedy.nodes_updated == 0  # Greedy never updates nodes.


def test_order():
    state = Grid2D.State((0, 0))

    node_1 = Greedy.GreedyNode(state, action=None, parent=None, h=1)
    node_2 = Greedy.GreedyNode(state, action=None, parent=None, h=2)

    assert node_1 < node_2
