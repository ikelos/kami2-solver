import sys

import informed_search
import statespace as kami2
from image_parser import parse_image_graph


def solve(puzzle_img, num_colors, puzzle_moves_left):
    puzzle_graph, puzzle_node_colors, pixel_nodes, label_names = parse_image_graph(puzzle_img, num_colors,
                                                                                   debug_plots = True)

    actions = astarSolver(puzzle_graph, puzzle_moves_left, puzzle_node_colors)
    print()
    print("SOLUTION:")
    for action in actions:
        print("Turn node {} to PALETTE {}".format(action[0], label_names[action[1]]))


def astarSolver(puzzle_graph, puzzle_moves_left, puzzle_node_colors):
    puzzle_step0 = kami2.PuzzleState(puzzle_graph, puzzle_node_colors, puzzle_moves_left)
    puzzle = kami2.Kami2Puzzle(puzzle_step0)
    # print("Solving using DFS:")
    # search.DepthFirstSearch().solve(puzzle)
    #
    # print("Solving using UCS:")
    # search.UniformCostSearch().solve(puzzle)
    # print("Solving using A* (# colors heuristic):")
    # informed_search.AStarSearch(informed_search.num_colors_heuristic).solve(puzzle)
    print("Solving using A* (color distance heuristic):")
    solver = informed_search.AStarSearch(informed_search.color_distance_heuristic)
    solver.solve(puzzle)
    return solver.actions


if __name__ == "__main__":
    solve(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
