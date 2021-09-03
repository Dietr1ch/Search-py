from typing import List, Optional

from search.algorithms.bfs import BFS
from search.algorithms.search import Node, SearchAlgorithm
from search.problems.grid.board2d import Grid2D, Grid2DMetaProblem
from search.space import Problem, Space


def test_no_solution():
    metaproblem = Grid2DMetaProblem(
        [
            "S  ",
        ]
    )
    problem: Problem = next(iter(metaproblem.multi_goal_given()))
    bfs: SearchAlgorithm = BFS(problem)

    # Search
    goal_node: Optional[Node] = bfs.search()

    # There's no solution
    assert goal_node is None
    # This maps needs to be completely expanded
    assert bfs.expansions == 3


def test_walk_all_directions():
    metaproblems = [
        Grid2DMetaProblem(
            [
                "S G",
            ]
        ),
        Grid2DMetaProblem(
            [
                "G S",
            ]
        ),
        Grid2DMetaProblem(
            [
                "G",
                " ",
                "S",
            ]
        ),
        Grid2DMetaProblem(
            [
                "S",
                " ",
                "G",
            ]
        ),
    ]
    for mp in metaproblems:
        problem: Problem = next(iter(mp.multi_goal_given()))
        bfs: SearchAlgorithm = BFS(problem)

        # Search
        goal_node: Optional[Node] = bfs.search()

        # There's a solution
        assert goal_node is not None
        # We expanded the "whole" space.
        # NOTE(ddaroch): There's an off-by-1 because we check goals first
        assert bfs.expansions == 3 - 1

        # We can get its path
        path = goal_node.path(problem.space)
        assert path is not None

        # We can get its path
        assert path.cost() == 2
