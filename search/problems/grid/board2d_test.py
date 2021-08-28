from typing import List, Optional

from search.algorithms.bfs import BFS
from search.algorithms.search import Node, SearchAlgorithm
from search.problems.grid.board2d import Board2D
from search.space import Problem, SimpleProblem, Space


def test_no_solution():
    space: Space = Board2D([
        "S  ",
    ])
    problem: Problem = next(iter(SimpleProblem.multi_goal_given(space=space)))
    bfs: SearchAlgorithm = BFS(problem)

    # Search
    goal_node: Optional[Node] = bfs.search()

    # There's no solution
    assert goal_node is None
    # This maps needs to be completely expanded
    assert bfs.expansions == 3


def test_walk_all_directions():
    spaces: List[Space] = [
        Board2D([
            "S G",
        ]),
        Board2D([
            "G S",
        ]),
        Board2D([
            "G",
            " ",
            "S",
        ]),
        Board2D([
            "S",
            " ",
            "G",
        ]),
    ]
    for space in spaces:
        problem: Problem = next(iter(SimpleProblem.multi_goal_given(space=space)))
        bfs: SearchAlgorithm = BFS(problem)

        # Search
        goal_node: Optional[Node] = bfs.search()

        # There's a solution
        assert goal_node is not None
        # We expanded the "whole" space.
        # NOTE(ddaroch): There's an off-by-1 because we check goals first
        assert bfs.expansions == 3 - 1

        # We can get its path
        path = goal_node.path(space)
        assert path is not None

        # We can get its path
        assert path.cost() == 2
