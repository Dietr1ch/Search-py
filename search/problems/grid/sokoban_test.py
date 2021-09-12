from typing import Optional

from search.algorithms.bfs import BFS
from search.algorithms.search import Node, SearchAlgorithm
from search.problems.grid.sokoban import SokobanMetaProblem
from search.space import Problem

INFINITY = float("inf")


def test_no_solution():
    metaproblem = SokobanMetaProblem(
        [
            "S B",
        ]
    )
    problem: Problem = next(iter(metaproblem.multi_goal_given()))
    bfs: SearchAlgorithm = BFS(problem)

    # Search
    goal_node: Optional[Node] = bfs.search()

    # There's no solution
    assert goal_node is None
    # This maps needs to be completely expanded
    assert bfs.expansions == 2


def test_push_all_directions():
    metaproblems = [
        SokobanMetaProblem(
            [
                "S BG",
            ]
        ),
        SokobanMetaProblem(
            [
                "GB S",
            ]
        ),
        SokobanMetaProblem(
            [
                "G",
                "B",
                " ",
                "S",
            ]
        ),
        SokobanMetaProblem(
            [
                "S",
                " ",
                "B",
                "G",
            ]
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
        assert bfs.expansions == 3 - 1

        # We can get its path
        path = goal_node.path(problem.space)
        assert path.cost() == 2


def test_multi_goal():
    metaproblem = SokobanMetaProblem(
        [
            "###G##",
            "GBS!BG",
            "##  ##",
        ]
    )
    problem: Problem = next(iter(metaproblem.multi_goal_given()))
    bfs: SearchAlgorithm = BFS(problem)

    # Search
    goal_node: Optional[Node] = bfs.search()

    # There's a solution
    assert goal_node is not None
    # This is not so easy
    assert 10 < bfs.expansions < 40

    # We can get its path
    path = goal_node.path(problem.space)
    assert path.cost() == 6


def test_undo():
    """A puzzle that requires relocating a box already in a goal.

    It should be harder with Greedy."""
    metaproblem = SokobanMetaProblem(
        [
            "S ! G",
            " # # ",
            " #B# ",
            "  G  ",
        ]
    )
    problem: Problem = next(iter(metaproblem.multi_goal_given()))
    bfs: SearchAlgorithm = BFS(problem)

    # Search
    goal_node: Optional[Node] = bfs.search()

    # There's a solution
    assert goal_node is not None
    # This is not so easy
    assert 10 < bfs.expansions < 40

    # We can get its path
    path = goal_node.path(problem.space)
    assert path.cost() == 6


def test_heuristic_no_goal():
    metaproblem = SokobanMetaProblem(
        [
            "S  ",
        ]
    )
    problem: Problem = next(iter(metaproblem.multi_goal_given()))

    assert len(problem.starting_states) == 1
    for start in problem.starting_states:
        # pylint: protected-access
        assert problem._eval_heuristics(start) == {
            "SokobanBetterDistance": INFINITY,
            "SokobanSimpleManhattanDistance": INFINITY,
            "SokobanMisplacedBoxes": INFINITY,
            "SokobanDiscreteMetric": 0,
            "Heuristic": 0,
        }


def test_heuristic_single_goal():
    metaproblem = SokobanMetaProblem(
        [
            "S G",
        ]
    )
    problem: Problem = next(iter(metaproblem.multi_goal_given()))

    assert len(problem.starting_states) == 1
    for start in problem.starting_states:
        # pylint: protected-access
        assert problem._eval_heuristics(start) == {
            "SokobanBetterDistance": 0,
            "SokobanSimpleManhattanDistance": 0,
            "SokobanMisplacedBoxes": 0,
            "SokobanDiscreteMetric": 0,
            "Heuristic": 0,
        }


def test_heuristic_multi_goal():
    metaproblems = [
        SokobanMetaProblem(
            [
                "                 G",
                " S                ",
                "G                 ",
            ]
        ),
        SokobanMetaProblem(
            [
                "                 G",
                "                S ",
                "G                 ",
            ]
        ),
    ]

    for metaproblem in metaproblems:
        for problem in metaproblem.multi_goal_given():
            assert len(problem.starting_states) == 1
            for start in problem.starting_states:
                # pylint: protected-access
                assert problem._eval_heuristics(start) == {
                    "SokobanBetterDistance": 0,
                    "SokobanSimpleManhattanDistance": 0,
                    "SokobanMisplacedBoxes": 0,
                    "SokobanDiscreteMetric": 0,
                    "Heuristic": 0,
                }
