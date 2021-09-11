#!/usr/bin/env python

"""
A small binary showcasing the search library
"""

import random
from typing import List

import numpy as np  # type: ignore
from termcolor import colored

from search.algorithms.bfs import BFS
from search.algorithms.dfs import DFS
from search.algorithms.dijkstra import Dijkstra
from search.algorithms.greedy import Greedy
from search.algorithms.search import HeuristicSearchAlgorithm, SearchAlgorithm
from search.problems.grid.board2d import Grid2DMetaProblem, Grid2DProblem
from search.problems.grid.bomb import Bombs2DMetaProblem, Bombs2DProblem
from search.problems.nm_puzzle import NMPuzzleMetaProblem, NMPuzzleProblem
from search.space import Heuristic, Problem


def solve(algorithm_class, problem: Problem, heuristic_class):
    """Solves a problem with a given search algorithm.

    Returns: Dictionary with a summary and key metrics.
    """
    if issubclass(algorithm_class, HeuristicSearchAlgorithm):
        search_algorithm = algorithm_class(problem, heuristic_class(problem))
    else:
        search_algorithm = algorithm_class(problem)
    assert search_algorithm is not None

    goal_node = search_algorithm.search()

    assert search_algorithm.time_ns is not None
    time_ms = (search_algorithm.time_ns) / 1_000_000.0

    stats = {
        "summary": "No solution found for this problem after {} expansions".format(
            search_algorithm.expansions
        ),
        "cost": float("inf"),
        "length": float("inf"),
        "expansions": search_algorithm.expansions,
        "states_generated": search_algorithm.states_generated,
        "states_reached": search_algorithm.states_reached,
        "nodes_created": search_algorithm.nodes_created,
        "nodes_updated": search_algorithm.nodes_updated,
        "time_ms": time_ms,
    }
    if goal_node is None:
        stats[
            "summary"
        ] = "No solution found for this problem after {} expansions".format(
            search_algorithm.expansions
        )
        return stats

    path = goal_node.path(problem.space)

    stats["summary"] = "Expanded {:-3} nodes in t:{:.2}ms to find: {}".format(
        search_algorithm.expansions, time_ms, path
    )
    stats["path"] = path
    stats["cost"] = path.cost()
    stats["length"] = len(path)
    stats["actions"] = path.actions()
    return stats


def compare(algorithms: List[SearchAlgorithm], problem: Problem, heuristic_class):
    """Solves a problem with many search algorithms and compares the solutions.

    Returns: Dictionary with a summary and key metrics.
    """
    print(
        "Solving this {} problem with the '{}' heuristic,".format(
            problem.space.__class__.__name__, heuristic_class
        )
    )
    print(problem.start_to_str())

    solutions = dict()
    best = {
        "cost": float("inf"),
        "length": float("inf"),
        "expansions": float("inf"),
        "states_generated": float("inf"),
        "states_reached": float("inf"),
        "nodes_created": float("inf"),
        "nodes_updated": float("inf"),
        "time_ms": float("inf"),
    }
    metrics = list(best.keys())

    for algorithm in algorithms:
        solutions[algorithm] = solve(algorithm, problem, heuristic_class)
        for metric in metrics:
            if solutions[algorithm][metric] < best[metric]:
                best[metric] = solutions[algorithm][metric]

    metrics.remove("cost")
    for a, solution in solutions.items():
        print("  * {:20}: {}".format(a.name(), solution["summary"]))
        if solution["cost"] > best["cost"]:
            print(
                "    -",
                colored("Sub-optimal!!", "red", attrs=[]),
                " {} ({} vs {})".format(
                    colored(
                        "{:.2%}".format(solution["cost"] / best["cost"]),
                        "red",
                        attrs=["bold"],
                    ),
                    solution["cost"],
                    best["cost"],
                ),
            )
        for metric in metrics:
            if solution[metric] > best[metric]:
                if best[metric] > 0:
                    ratio = solution[metric] / best[metric]
                else:
                    ratio = float("inf")

                color = None
                attrs = []
                if ratio < 1.04:
                    continue

                if ratio >= 1.5:
                    color = "red"
                    attrs = ["bold"]
                elif ratio >= 1.2:
                    color = "yellow"
                    attrs = ["bold"]
                elif ratio >= 1.1:
                    color = "white"
                    attrs = ["bold"]
                print(
                    "    - Not the best on {}!! {} ({} vs {})".format(
                        colored("{:12}".format(metric), color, attrs=attrs),
                        colored(
                            "{:.2%}".format(ratio),
                            color,
                            attrs=attrs,
                        ),
                        solution[metric],
                        best[metric],
                    )
                )
    print("")
    return solutions


def main():
    """A simple program solving an easy maze."""
    metaproblems_dict = {
        Grid2DMetaProblem: {
            "problems": [
                Grid2DMetaProblem(
                    [
                        "   G ",
                        " ####",
                        "     ",
                        "#### ",
                        "     ",
                        "S    ",
                    ]
                ),
                Grid2DMetaProblem(
                    [
                        "G          ",
                        "           ",
                        "########## ",
                        "           ",
                        "          G",
                        " ##########",
                        "           ",
                        "           ",
                        "########## ",
                        "           ",
                        "S          ",
                    ]
                ),
                # It can't get easier right?
                Grid2DMetaProblem(
                    [
                        "G  S{:60}G".format(" "),
                    ]
                ),
                # What if there's no goal?
                Grid2DMetaProblem(
                    [
                        "   S{:60} ".format(" "),
                    ]
                ),
            ],
            "heuristics": Grid2DProblem.all_heuristics(),
        },
        Bombs2DMetaProblem: {
            "problems": [
                Bombs2DMetaProblem(
                    [
                        "  G",
                        "###",
                        "B S",
                    ],
                    starting_bombs=0,
                ),
                Bombs2DMetaProblem(
                    [
                        "G    ",
                        "#####",
                        "    B",
                        "S    ",
                        "B    ",
                    ],
                    starting_bombs=0,
                ),
            ],
            "heuristics": Bombs2DProblem.all_heuristics(),
        },
        NMPuzzleMetaProblem: {
            "problems": [
                # NOTE(ddaroch): The puzzles from these states to random goals might
                # be unexpected.
                NMPuzzleMetaProblem(
                    np.array(
                        [
                            [1, 2, 0],
                        ]
                    )
                ),
                NMPuzzleMetaProblem(
                    np.array(
                        [
                            [0, 1, 2],
                            [3, 4, 5],
                            [8, 6, 7],
                        ]
                    )
                ),
                NMPuzzleMetaProblem(
                    np.array(
                        [
                            [11, 1, 2],
                            [0, 4, 5],
                            [3, 7, 8],
                            [6, 9, 10],
                        ]
                    )
                ),
            ],
            "heuristics": NMPuzzleProblem.all_heuristics(),
        },
    }

    random_problems = 1

    # pylint: disable=invalid-name

    algorithms = [
        DFS,
        BFS,
        Dijkstra,
        Greedy,
    ]

    for problem_class in metaproblems_dict:
        metaproblems = metaproblems_dict[problem_class]["problems"]
        heuristic_classes = metaproblems_dict[problem_class]["heuristics"]
        if len(heuristic_classes) == 0:
            heuristic_classes = [Heuristic]

        # Generate all the problems for this problem class
        problems = []
        for mp in metaproblems:
            # Add all the simple given problems
            for p in mp.simple_given():
                problems.append(p)

            # Add all the multi-goal given problems
            for p in mp.multi_goal_given():
                problems.append(p)

            random.seed(1)
            for _ in range(random_problems):
                problems.append(mp.simple_random())

        for problem in problems:
            for heuristic_class in heuristic_classes:
                compare(algorithms, problem, heuristic_class)


if __name__ == "__main__":
    main()
