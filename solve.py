import sys

import informed_search
import statespace as kami2
from image_parser import parse_image_graph


def solve(puzzle_img, num_colors, puzzle_moves_left):
    puzzle_graph, puzzle_node_colors, pixel_nodes = parse_image_graph(puzzle_img, num_colors,
                                                                      debug_plots = True)

    actions = astarSolver(puzzle_graph, puzzle_moves_left, puzzle_node_colors)

    # oldSolver(puzzle_graph, puzzle_moves_left, puzzle_node_colors)


# def create_regions(puzzle_graph, puzzle_node_colors) -> Tuple[List[Region], List[str]]:
#     output_regions = []
#     output_colors = list(set(puzzle_node_colors.values()))
#     for node_num in puzzle_graph:
#         output_regions.append(
#             Region(str(int(node_num)), output_colors.index(puzzle_node_colors[node_num]),
#                    set([str(int(x)) for x in puzzle_graph[node_num]])))
#     output_colors = ["COLOR {}".format(x) for x in output_colors]
#     return output_regions, output_colors

# def oldSolver(puzzle_graph, puzzle_moves_left, puzzle_node_colors):
#     regions, colors = create_regions(puzzle_graph, puzzle_node_colors)
#     kami = Kami(regions, colors)
#     print("Solving with the knowledge that optimal number of moves is {}:".format(puzzle_moves_left))
#     kami.solve(puzzle_moves_left)


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
