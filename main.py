#!/usr/bin/env python

"""
A small binary showcasing the search library
"""

import random
import time
from typing import List

from termcolor import colored

from search.algorithms.bfs import BFS
from search.algorithms.dfs import DFS
from search.algorithms.search import SearchAlgorithm
from search.problems.grid.board2d import Board2D
from search.space import Problem


def solve(algorithm_class, problem: Problem):
    """Solves a problem with a given search algorithm.

    Returns: Dictionary with a summary and key metrics.
    """
    search_algorithm: SearchAlgorithm = algorithm_class(problem)
    start_ts_ns = time.perf_counter_ns()
    goal_node = search_algorithm.search()
    end_ts_ns = time.perf_counter_ns()
    time_ms = (end_ts_ns - start_ts_ns) / 1_000_000.0
    if goal_node is None:
        return {
            "summary": "No solution found for this problem after {} expansions".format(search_algorithm.expansions),
            "cost": float("inf"),
            "length": float("inf"),
            "expansions": search_algorithm.expansions,
            "time_ms": time_ms,
        }

    start, actions, goal = goal_node.path()

    compressed_path = []
    if len(actions):
        last = actions[0]
        count = 0

        # pylint: disable=invalid-name
        for a in actions:
            if a == last:
                count += 1
                continue
            compressed_path.append("%2d%s" % (count, str(last)))
            count = 1
            last = a
        compressed_path.append("%2d%s" % (count, str(last)))

    cost = sum([a.cost() for a in actions])
    length = len(actions)

    return {
        "summary": "expanded {:-3} nodes to find: (c:{}, l:{}, t:{:.2}ms) {} => {} => {}".format(
            search_algorithm.expansions,
            cost,
            length,
            time_ms,
            colored(str(start), 'green', attrs=['bold']),
            colored(" ".join(compressed_path), 'blue', attrs=['bold']),
            colored(str(goal), 'yellow', attrs=['bold'])),
        "cost": cost,
        "length": length,
        "expansions": search_algorithm.expansions,
        "time_ms": time_ms,
    }


def compare(algorithms: List[SearchAlgorithm], problem: Problem):
    """Solves a problem with many search algorithms and compares the solutions.

    Returns: Dictionary with a summary and key metrics.
    """
    print("Solving this %s problem," % problem.space.__class__.__name__)
    print(board2d_str(problem))

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
                elif ratio >= 1.5:
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


def board2d_str(board2d_problem):
    """Formats a Problem over a Board2D to an ASCII colored string.

    Drawing a board is not enough as we want the problem's start and goal
    states.
    We don't have a nice way to write a generic function printing a generic
    problem based on a generic space printer.
    """
    space = board2d_problem.space

    board_str = ''
    board_str += colored(('    █' + ('█' * (space.W)) +
                         '█\n'), 'green', attrs=['bold'])

    for y, row in enumerate(space.grid):
        board_str += colored("%3d " % y, 'white')
        board_str += colored("█", 'green', attrs=['bold'])
        for x, is_wall in enumerate(row):
            if board2d_problem.is_goal((x, y)):
                board_str += colored('G', 'yellow', attrs=['bold'])
            elif (x, y) in board2d_problem.starts:
                board_str += colored('S', 'white', attrs=['bold'])
            elif is_wall:
                board_str += colored('█', 'green', attrs=['bold'])
            else:
                board_str += " "
        board_str += colored('█', 'green', attrs=['bold']) + '\n'

    board_str += colored(('    █' + ('█' * (space.W)) +
                         '█\n'), 'green', attrs=['bold'])

    return board_str


def main():
    """A simple program solving an easy maze."""
    spaces = [
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
        # Add all the simple given problems
        for p in space.simple_given():
            problems.append(p)

        # Add all the multi-goal given problems
        for p in space.multi_goal_given():
            problems.append(p)

        random.seed(1)
        for _ in range(random_problems):
            problems.append(space.simple_random())

    algorithms = [
        DFS,
        BFS,
    ]

    for p in problems:
        compare(algorithms, p)


if __name__ == "__main__":
    main()
