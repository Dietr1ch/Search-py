#!/usr/bin/env python

"""
A small binary showcasing the search library
"""

import random
from typing import List

from termcolor import colored

from search.algorithms.astar import AStar
from search.algorithms.bfs import BFS
from search.algorithms.dfs import DFS
from search.algorithms.dijkstra import Dijkstra
from search.algorithms.search import HeuristicSearchAlgorithm, SearchAlgorithm
from search.problems.grid.board2d import Grid2DMetaProblem
from search.problems.grid.bomb import Bombs2DMetaProblem
from search.space import Heuristic, Problem, ZeroHeuristic


def solve(algorithm_class, problem: Problem, heuristic: Heuristic):
    """Solves a problem with a given search algorithm.

    Returns: Dictionary with a summary and key metrics.
    """
    if issubclass(algorithm_class, HeuristicSearchAlgorithm):
        search_algorithm: HeuristicSearchAlgorithm = algorithm_class(problem, heuristic)
    elif issubclass(algorithm_class, SearchAlgorithm):
        search_algorithm: SearchAlgorithm = algorithm_class(problem)
    else:
        raise ValueError("")
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


def compare(algorithms: List[SearchAlgorithm], problem: Problem):
    """Solves a problem with many search algorithms and compares the solutions.

    Returns: Dictionary with a summary and key metrics.
    """
    print("Solving this %s problem," % problem.space.__class__.__name__)
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

    for a in algorithms:
        heuristics = [ZeroHeuristic(problem)]
        if issubclass(a, HeuristicSearchAlgorithm):
            heuristics = problem.all_heuristics()

        for h in heuristics:
            solutions[(a, h)] = solve(a, problem, h)
            for metric in metrics:
                if solutions[(a, h)][metric] < best[metric]:
                    best[metric] = solutions[(a, h)][metric]

    for (a, h), solution in solutions.items():
        # Algorithm + Heuristic name
        if issubclass(a, HeuristicSearchAlgorithm):
            print("  * {} + {}".format(a.name(), h))
        else:
            print("  * {}".format(a.name()))

        # Summary
        print("    {}".format(solution["summary"]))

        # Check optimality
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

        # Print all metrics
        for metric in metrics:
            try:
                ratio = solution[metric] / best[metric]
            except ZeroDivisionError:
                ratio = float('inf')

            if ratio < 1.0:
                raise ValueError(f"What's going on with metric '{metric}'?")
            elif solution[metric] == best[metric]:
                comment = "The best "
                color = "green"
                attrs = ["bold"]
            elif ratio <= 1.15:
                comment = "Not bad  "
                color = "white"
                attrs = ["bold"]
            elif ratio <= 1.4:
                comment = "Not awful"
                color = "yellow"
                attrs = ["bold"]
            else:
                comment = "Terrible "
                color = "red"
                attrs = ["bold"]

            print(
                "    - {} on {}!! {} ({} vs {})".format(
                    comment,
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
    metaproblems = [
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
    ]

    problems = []
    random_problems = 1

    # pylint: disable=invalid-name
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

    algorithms = [
        DFS,
        BFS,
        Dijkstra,
        AStar,
    ]

    for p in problems:
        compare(algorithms, p)


if __name__ == "__main__":
    main()
