"""
Tests for the NM-puzzle. A generalization of the 8-puzzle

An implementation detail that could be surprising, is that the empty space uses
the highest number instead of 0.
"""
from typing import Optional

import numpy as np
import pytest
from search.algorithms.bfs import BFS
from search.algorithms.search import Node, SearchAlgorithm
from search.problems.nm_puzzle import NMPuzzle, NMPuzzleMetaProblem
from search.space import Heuristic, Problem

INFINITY = float("inf")


def test_no_solution():
    metaproblem = NMPuzzleMetaProblem(
        np.array(
            [
                [1, 0, 2],
            ]
        )
    )

    problem: Problem = next(iter(metaproblem.simple_given()))
    bfs: SearchAlgorithm = BFS(problem)

    # Search
    goal_node: Optional[Node] = bfs.search()

    # There's no solution
    assert goal_node is None
    # This maps needs to be completely expanded
    assert bfs.expansions == 3


def test_no_solution_tile_swap():
    """The search space is not connected."""
    metaproblem = NMPuzzleMetaProblem(
        np.array(
            [
                [1, 0],
                [2, 3],
            ]
        )
    )

    problem: Problem = next(iter(metaproblem.simple_given()))
    bfs: SearchAlgorithm = BFS(problem)

    # Search
    goal_node: Optional[Node] = bfs.search()

    # There's no solution
    assert goal_node is None
    assert 10 < bfs.expansions < 20


def test_move_up_left():
    metaproblems = [
        NMPuzzleMetaProblem(
            np.array(
                [
                    [0, 1],
                    [3, 2],  # <-
                ]
            )
        ),
        NMPuzzleMetaProblem(
            np.array(
                [
                    [0, 3],  # ^
                    [2, 1],
                ]
            )
        ),
    ]

    # pylint: disable=invalid-name
    for mp in metaproblems:
        problem: Problem = next(iter(mp.simple_given()))
        bfs: SearchAlgorithm = BFS(problem)

        # Search
        goal_node: Optional[Node] = bfs.search()

        # There's a solution
        assert goal_node is not None
        # We expanded the "whole" space.
        # NOTE(ddaroch): There's an off-by-1 because we check goals first
        assert 1 <= bfs.expansions < 3

        # We can get its path
        path = goal_node.path(problem.space)
        assert path is not None

        # We can get its path
        assert path.cost() == 1


def test_spin():
    metaproblem = NMPuzzleMetaProblem(
        np.array(
            [
                [2, 3],
                [1, 0],
            ]
        )
    )

    problem: Problem = next(iter(metaproblem.simple_given()))
    bfs: SearchAlgorithm = BFS(problem)

    # Search
    goal_node: Optional[Node] = bfs.search()

    # There's a solution
    assert goal_node is not None
    # We expanded the "whole" space.
    # NOTE(ddaroch): There's an off-by-1 because we check goals first
    assert 4 <= bfs.expansions < 20

    # We can get its path
    path = goal_node.path(problem.space)
    assert path is not None

    # We can get its path
    assert path.cost() == 5
    # And its actions
    assert path.actions() == [
        # NOTE: Spinning clockwise needs 7 actions. BFS won't find it.
        NMPuzzle.Action.UP,
        NMPuzzle.Action.RIGHT,
        NMPuzzle.Action.DOWN,
        NMPuzzle.Action.LEFT,
        NMPuzzle.Action.UP,
    ]


@pytest.mark.skip(reason="NMPuzzleManhattanDistance is not implemented yet.")
def test_heuristic_single_tile_off():
    metaproblem = NMPuzzleMetaProblem(
        np.array(
            [
                [0, 1],
                [3, 2],  # <-
            ]
        )
    )
    problem: Problem = next(iter(metaproblem.simple_given()))

    assert len(problem.starting_states) == 1
    for start in problem.starting_states:
        h_values = [h(start) for h in problem.all_heuristics()]
        assert h_values == [1, 0]


@pytest.mark.skip(reason="NMPuzzleManhattanDistance is not implemented yet.")
def test_heuristic_reversed_tiles():
    metaproblem = NMPuzzleMetaProblem(
        np.array(
            [
                # Is this solvable? Well, it should seem hard to solve anyways
                [8, 7, 6],
                [5, 4, 3],
                [2, 1, 0],
            ]
        )
    )
    problem: Problem = next(iter(metaproblem.simple_given()))

    assert len(problem.starting_states) == 1
    for start in problem.starting_states:
        h_values = [h(start) for h in problem.all_heuristics()]
        assert h_values == [24, 0]
