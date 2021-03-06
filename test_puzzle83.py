import statespace as kami2
import informed_search

puzzle83_graph = {
    1: frozenset([2, 3]),
    2: frozenset([1, 3, 4]),
    3: frozenset([1, 2, 25]),
    4: frozenset([2, 5, 6]),
    5: frozenset([4, 6, 7]),
    6: frozenset([4, 5]),
    7: frozenset([5, 8, 9]),
    8: frozenset([7, 9]),
    9: frozenset([7, 8, 10]),
    10: frozenset([9, 11, 12]),
    11: frozenset([10, 12, 13, 15, 52]),
    12: frozenset([10, 11, 19, 33]),
    13: frozenset([11, 14, 15]),
    14: frozenset([13, 15]),
    15: frozenset([11, 13, 14, 16]),
    16: frozenset([15, 17, 18]),
    17: frozenset([16, 18]),
    18: frozenset([16, 17, 22]),
    19: frozenset([12, 20, 21]),
    20: frozenset([19, 21]),
    21: frozenset([19, 20, 22]),
    22: frozenset([18, 21, 23, 24]),
    23: frozenset([22, 24]),
    24: frozenset([22, 23]),
    25: frozenset([3, 26, 27]),
    26: frozenset([25, 27]),
    27: frozenset([25, 26, 28]),
    28: frozenset([27, 29, 30, 31]),
    29: frozenset([28,]),
    30: frozenset([28, 31, 34]),
    31: frozenset([28, 30, 35]),
    32: frozenset([33, 34]),
    33: frozenset([12, 32, 34]),
    34: frozenset([30, 32, 33, 35, 36]),
    35: frozenset([31, 34, 36, 37]),
    36: frozenset([34, 35, 37]),
    37: frozenset([35, 36, 38]),
    38: frozenset([37, 39, 40]),
    39: frozenset([38, 40, 41]),
    40: frozenset([38, 39, 44]),
    41: frozenset([39, 42, 43]),
    42: frozenset([41, 43]),
    43: frozenset([41, 42, 45, 47]),
    44: frozenset([40, 45, 46]),
    45: frozenset([43, 44, 46, 47]),
    46: frozenset([44, 45, 47, 48]),
    47: frozenset([43, 45, 46, 48]),
    48: frozenset([46, 47, 49]),
    49: frozenset([48, 50, 51]),
    50: frozenset([49, 51]),
    51: frozenset([49, 50]),
    52: frozenset([11,])
}
puzzle83_node_colors = {
    1: 'g',
    2: 'm',
    3: 'r',
    4: 'g',
    5: 'm',
    6: 'r',
    7: 'g',
    8: 'm',
    9: 'r',
    10: 'g',
    11: 'm',
    12: 'r',
    13: 'g',
    14: 'm',
    15: 'r',
    16: 'g',
    17: 'm',
    18: 'r',
    19: 'g',
    20: 'm',
    21: 'r',
    22: 'g',
    23: 'm',
    24: 'r',
    25: 'g',
    26: 'm',
    27: 'r',
    28: 'g',
    29: 'm',
    30: 'm',
    31: 'r',
    32: 'g',
    33: 'm',
    34: 'r',
    35: 'g',
    36: 'm',
    37: 'r',
    38: 'g',
    39: 'm',
    40: 'r',
    41: 'g',
    42: 'm',
    43: 'r',
    44: 'g',
    45: 'm',
    46: 'r',
    47: 'g',
    48: 'm',
    49: 'g',
    50: 'm',
    51: 'r',
    52: 'g'
}
puzzle83_moves_left = 10

puzzle83_step0 = kami2.PuzzleState(puzzle83_graph, puzzle83_node_colors, puzzle83_moves_left)
puzzle83 = kami2.Kami2Puzzle(puzzle83_step0)

def main():
    # print("Solving using DFS:")
    # search.DepthFirstSearch().solve(puzzle83)

    # print("Solving using A* (# colors heuristic):")
    # informed_search.AStarSearch(informed_search.num_colors_heuristic).solve(puzzle83)

    print("Solving using A* (color distance heuristic):")
    informed_search.AStarSearch(informed_search.color_distance_heuristic).solve(puzzle83)

if __name__ == "__main__":
    main()
