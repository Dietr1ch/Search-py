#!/usr/bin/env python

"""
A small binary showcasing the search library
"""

import random
from typing import List

from termcolor import colored

from search.algorithms.bfs import BFS
from search.algorithms.dfs import DFS
from search.algorithms.search import SearchAlgorithm
from search.problems.grid.board2d import Board2D
from search.space import (PredefinedSpace, Problem, RandomAccessSpace,
                          SimpleProblem, Space)


def solve(algorithm_class, problem: Problem):
    """Solves a problem with a given search algorithm.

    Returns: Dictionary with a summary and key metrics.
    """
    search_algorithm: SearchAlgorithm = algorithm_class(problem)
    goal_node = search_algorithm.search()
    time_ms = (search_algorithm.time_ns) / 1_000_000.0
    if goal_node is None:
        return {
            "summary": "No solution found for this problem after {} expansions".format(search_algorithm.expansions),
            "cost": float("inf"),
            "length": float("inf"),
            "expansions": search_algorithm.expansions,
            "time_ms": time_ms,
        }

    path = goal_node.path(problem.space)

    return {
        "summary": "Expanded {:-3} nodes in t:{:.2}ms to find: {}".format(
            search_algorithm.expansions,
            time_ms,
            path),
        "path": path,
        "cost": path.cost(),
        "length": len(path),
        "expansions": search_algorithm.expansions,
        "time_ms": time_ms,
        "actions": path.actions(),
    }


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
        "time_ms": float("inf"),
    }
    metrics = list(best.keys())

    for a in algorithms:
        solutions[a] = solve(a, problem)
        for metric in metrics:
            if solutions[a][metric] < best[metric]:
                best[metric] = solutions[a][metric]

    for a in solutions:
        print("  * {:20}: {}".format(a.name(), solutions[a]["summary"]))
        for metric in metrics:
            if solutions[a][metric] > best[metric]:
                ratio = solutions[a][metric] / best[metric]
                color = None
                attrs = []
                if ratio < 1.04:
                    continue

                if ratio >= 1.5:
                    color = 'red'
                    attrs = ['bold']
                elif ratio >= 1.2:
                    color = 'yellow'
                    attrs = ['bold']
                elif ratio >= 1.1:
                    color = 'white'
                    attrs = ['bold']
                print("    - Not the best on {}!! {} ({} vs {})".format(
                    colored("{:12}".format(metric), color, attrs=attrs),
                    colored("{:.2%}".format(solutions[a][metric] / best[metric]),
                            color, attrs=attrs),
                    solutions[a][metric],
                    best[metric]))
    print("")
    return solutions


def main():
    """A simple program solving an easy maze."""
    spaces: List[Space] = [
        Board2D([
            "   G ",
            " ####",
            "     ",
            "#### ",
            "     ",
            "S    ",
        ]),
        Board2D([
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
        ]),
        # It can't get easier right?
        Board2D([
            "G  S{:60}G".format(" "),
        ]),
        # What if there's no goal?
        Board2D([
            "   S{:60} ".format(" "),
        ]),
    ]

    # pylint: disable=invalid-name

    problems = []
    random_problems = 1
    for space in spaces:
        if isinstance(space, PredefinedSpace):
            # Add all the simple given problems
            for p in SimpleProblem.simple_given(space=space):
                problems.append(p)

            # Add all the multi-goal given problems
            for p in SimpleProblem.multi_goal_given(space=space):
                problems.append(p)

        if isinstance(space, RandomAccessSpace):
            random.seed(1)
            for _ in range(random_problems):
                problems.append(SimpleProblem.simple_random(space))

    algorithms = [
        DFS,
        BFS,
    ]

    for p in problems:
        compare(algorithms, p)


if __name__ == "__main__":
    main()
