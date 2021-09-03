from typing import Optional

from search.algorithms.dfs import DFS
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
    dfs: SearchAlgorithm = DFS(problem)

    # Search
    goal_node: Optional[Node] = dfs.search()

    assert goal_node is None
    assert 18 <= dfs.expansions <= 22
    assert 4 < dfs.time_ns < 10_000_000


def test_expansion_order():
    length = 1000

    # A long aisle with the goal on one end.
    # DFS should get lucky on one and head straight to the goal, and get
    # unlucky on the other one and expand all the nodes.
    # This holds because the expansion order and retrieval from Open is
    # deterministic.
    empty_space_str = "" + " " * length + "S" + " " * length
    metaproblems = [
        Grid2DMetaProblem([
            "G" + empty_space_str,
        ]),
        Grid2DMetaProblem([
            empty_space_str + "G",
        ]),
    ]

    solutions = []
    for mp in metaproblems:
        problem: Problem = next(iter(mp.multi_goal_given()))
        dfs: SearchAlgorithm = DFS(problem)

        # Search
        goal_node: Optional[Node] = dfs.search()
        assert goal_node is not None
        assert goal_node is not None

        path = goal_node.path(problem.space)
        assert path is not None

        solutions.append({
            "algorithm": dfs,
            "goal_node": goal_node,
            "path": path,
        })

    lengths = [len(path) for _, _, path in solutions]
    if lengths[0] > lengths[1]:
        solutions.reverse()

    good_luck = solutions[0]
    bad_luck = solutions[1]
    assert len(good_luck["path"]) == length + 1
    assert good_luck["algorithm"].expansions == 2 * length + 1

    assert len(bad_luck["path"]) == length + 1
    assert bad_luck["algorithm"].expansions == length + 1
