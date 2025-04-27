from typing import Optional

from search.algorithms.bfs import BFS
from search.algorithms.search import Node, SearchAlgorithm
from search.problems.grid.bomb import Bombs2DMetaProblem
from search.space import Problem

INFINITY = float("inf")


def test_no_solution():
    metaproblem = Bombs2DMetaProblem(
        [
            "S  ",
        ],
        starting_bombs=1,
    )
    problem: Problem = next(iter(metaproblem.multi_goal_given()))
    bfs: SearchAlgorithm = BFS(problem)

    # Search
    goal_node: Optional[Node] = bfs.search()

    # There's no solution
    assert goal_node is None
    # This maps needs to be completely expanded
    assert bfs.expansions == 3


def test_walk_and_bomb_all_directions():
    metaproblems = [
        Bombs2DMetaProblem(
            [
                "S#G",
            ],
            starting_bombs=1,
        ),
        Bombs2DMetaProblem(
            [
                "G#S",
            ],
            starting_bombs=1,
        ),
        Bombs2DMetaProblem(
            [
                "G",
                "#",
                "S",
            ],
            starting_bombs=1,
        ),
        Bombs2DMetaProblem(
            [
                "S",
                "#",
                "G",
            ],
            starting_bombs=1,
        ),
    ]

    # pylint: disable=invalid-name
    for mp in metaproblems:
        problem: Problem = next(iter(mp.multi_goal_given()))
        bfs: SearchAlgorithm = BFS(problem)

        # Search
        goal_node: Optional[Node] = bfs.search()

        # There's a solution
        assert goal_node is not None
        # We expanded the "whole" space.
        # NOTE(ddaroch): There's an off-by-1 because we check goals first
        assert bfs.expansions == 4 - 1

        # We can get its path
        path = goal_node.path(problem.space)
        assert path is not None

        # We can get its path
        assert path.cost() == 3


def test_heuristic_no_goal():
    metaproblem = Bombs2DMetaProblem(
        [
            "S  ",
        ],
        starting_bombs=1,
    )
    problem: Problem = next(iter(metaproblem.multi_goal_given()))

    assert len(problem.starting_states) == 1
    for start in problem.starting_states:
        h_values = [h(start) for h in problem.all_heuristics()]
        assert h_values == [INFINITY, INFINITY, 1, 0]


def test_heuristic_single_goal():
    metaproblem = Bombs2DMetaProblem(
        [
            "S G",
        ],
        starting_bombs=1,
    )
    problem: Problem = next(iter(metaproblem.multi_goal_given()))

    assert len(problem.starting_states) == 1
    for start in problem.starting_states:
        h_values = [h(start) for h in problem.all_heuristics()]
        assert h_values == [2, 2, 1, 0]


def test_heuristic_multi_goal():
    metaproblems = [
        Bombs2DMetaProblem(
            [
                "                 G",
                " S                ",
                "G                 ",
            ],
            starting_bombs=1,
        ),
        Bombs2DMetaProblem(
            [
                "                 G",
                "                S ",
                "G                 ",
            ],
            starting_bombs=1,
        ),
    ]

    for metaproblem in metaproblems:
        for problem in metaproblem.multi_goal_given():
            assert len(problem.starting_states) == 1
            for start in problem.starting_states:
                h_values = [h(start) for h in problem.all_heuristics()]
                assert h_values == [2, 1, 1, 0]
