import argparse
import collections
import copy
import itertools

import cv2
import numpy as np
from matplotlib import pyplot as plt

# TODO I know this is hardcoded, but the screenshots should always be the same resolution (750x1334), right?
puzzle_height_y = 1210
puzzle_width_x = 750


def get_original_image(filename):
    return cv2.imread(filename, cv2.IMREAD_COLOR)


def image_preprocessing(img):
    bilateral_d = 5
    bilateral_sigma = 100
    bilateral_filtered_img = cv2.bilateralFilter(img, bilateral_d, bilateral_sigma, bilateral_sigma)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("original image")

    ax2.imshow(cv2.cvtColor(bilateral_filtered_img, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"bilateral filter (d={bilateral_d}, sigma={bilateral_sigma})")
    # plt.show()
    return bilateral_filtered_img


def crop_to_puzzle(full_screenshot_img):
    lines = get_lines(full_screenshot_img)
    horizontal_lines_ys = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 3:
            horizontal_lines_ys.append(y1)

    global puzzle_height_y, puzzle_width_x
    puzzle_height_y = max(horizontal_lines_ys) - min(horizontal_lines_ys)
    puzzle_width_x = full_screenshot_img.shape[1]
    puzzle = copy.deepcopy(full_screenshot_img)
    return puzzle[min(horizontal_lines_ys):max(horizontal_lines_ys)]


def get_lines(full_screenshot_img, min_line_length = 100, max_line_gap = 10):
    gray = cv2.cvtColor(full_screenshot_img, cv2.COLOR_BGR2RGB)
    edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
    minLineLength = min_line_length
    maxLineGap = max_line_gap
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    return lines


def crop_to_color_palette(full_screenshot_img):
    img_copy = copy.deepcopy(full_screenshot_img)

    # top left corner of the color palette in the image (i.e. the node colors)
    lines = get_lines(full_screenshot_img, 100, 5)
    vertical_lines_xs = []
    horizontal_lines_ys = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 3:
            horizontal_lines_ys.append(y1)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if y2 > max(horizontal_lines_ys):
            if abs(x2 - x1) < 3:
                vertical_lines_xs.append(x1)

    banding_dividers = [vertical_lines_xs[0]]
    for line in sorted(vertical_lines_xs):
        if abs(line - banding_dividers[-1]) > 10:
            banding_dividers.append(line)

    start_y = max(horizontal_lines_ys)
    start_x = banding_dividers[4]
    height_y = full_screenshot_img.shape[0] - puzzle_height_y
    width_x = full_screenshot_img.shape[1] - banding_dividers[4]
    return img_copy[start_y:start_y + height_y, start_x:start_x + width_x]


def convert_to_kmeans_colors(labels, centers):
    flatten_labels = np.ravel(labels)
    # print("flattened labels:")
    # print(flatten_labels[0:5])

    # convert colors in puzzle to the k-means colors
    converted_pixels = [[round(centers[i, 0]), round(centers[i, 1]), round(centers[i, 2])] for i in flatten_labels]
    converted_puzzle = np.reshape(converted_pixels, (puzzle_height_y, puzzle_width_x, 3))
    return np.asarray(converted_puzzle, dtype = np.uint16)


def assign_pixels_to_nodes(pixel_labels, num_labels, ignore_labels = None):
    if ignore_labels is None:
        ignore_labels = []

    # parse contiguous regions (i.e. nodes) from converted colors
    pixel_nodes = np.zeros((puzzle_height_y, puzzle_width_x))
    next_node_number = 1

    masks_by_label = np.zeros((num_labels, pixel_labels.shape[0], pixel_labels.shape[1]))

    erosion_kernel = np.ones((5, 5), dtype = np.uint8)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    pixel_labels_ndarray = np.array(pixel_labels)

    for label_num in range(num_labels):
        # construct binary bitmask of the pixel_labels (1 for each pixel of that label, 0 otherwise)
        masks_by_label[label_num] = np.where(pixel_labels_ndarray == label_num, 1, 0)

        # erode by 3x3 square kernel, and then dilate by a 3x3 cross kernel to leave
        # a thin strip of pixels between neighboring regions
        erosion = cv2.erode(masks_by_label[label_num], erosion_kernel, iterations = 1)
        dilation = cv2.dilate(erosion, dilation_kernel, iterations = 1)

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1.imshow(masks_by_label[label_num], cmap = 'gray')
        # ax1.set_title(f"initial mask for label {label_num}")
        #
        # ax2.imshow(erosion, cmap = 'gray')
        # ax2.set_title(f"label {label_num} mask after erosion")
        #
        # ax3.imshow(dilation, cmap = 'gray')
        # ax3.set_title(f"label {label_num} mask after erosion & dilation")
        plt.show()

        # label each contiguous region in the resulting mask as a separate node
        for pixel_y in range(puzzle_height_y):
            for pixel_x in range(puzzle_width_x):
                # if pixel is not in the dilated mask or is already labeled, skip it
                if dilation[pixel_y, pixel_x] == 0 or pixel_nodes[pixel_y, pixel_x] > 0:
                    continue

                # print(f"new node {next_node_number}: starting with pixel (y={pixel_y}, x={pixel_x})")
                if pixel_labels[pixel_y, pixel_x] not in ignore_labels:
                    pixel_nodes[pixel_y, pixel_x] = next_node_number
                    next_node_number += 1
                else:
                    pixel_nodes[pixel_y, pixel_x] = -1

                neighbor_coords = get_neighbor_coords(pixel_y, pixel_x)
                # recursively assign the same node number to all contiguous pixels that are also in the dilation mask
                while len(neighbor_coords) > 0:
                    nbr_y, nbr_x = neighbor_coords.pop(0)
                    if pixel_nodes[nbr_y, nbr_x] == 0 and dilation[nbr_y, nbr_x] == 1:
                        pixel_nodes[nbr_y, nbr_x] = pixel_nodes[pixel_y, pixel_x]
                        neighbor_coords.extend(get_neighbor_coords(nbr_y, nbr_x))

    before_fill_in = pixel_nodes.copy()

    print("Assigning borders")

    # fill in pixels that weren't part of any mask (the pixels along the borders)
    any_updates = True
    while any_updates:
        tmp_pixel_nodes = pixel_nodes.copy()
        any_updates = False
        for pixel_y in range(puzzle_height_y):
            for pixel_x in range(puzzle_width_x):
                # if pixel is already labeled, skip it
                if tmp_pixel_nodes[pixel_y, pixel_x] != 0:
                    continue

                neighbor_coords = get_neighbor_coords(pixel_y, pixel_x)
                # if the labels on neighbors are all the same, use that label
                nbr_labels = []
                should_update = False  # update only if all labeled neighbors share the same label
                for nbr_y, nbr_x in neighbor_coords:
                    if pixel_nodes[nbr_y, nbr_x] != 0:
                        nbr_labels.append(pixel_nodes[nbr_y, nbr_x])

                should_update = True if len(nbr_labels) > 0 else False
                for nbr_label in nbr_labels:
                    if nbr_label != nbr_labels[0]:
                        should_update = False
                if should_update:
                    tmp_pixel_nodes[pixel_y, pixel_x] = nbr_labels[0]
                    any_updates = True
        pixel_nodes = tmp_pixel_nodes.copy()

    ## Debugging the fill-in process
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # c1 = ax1.pcolormesh(before_fill_in, cmap = 'jet')
    # ax1.invert_yaxis()
    # ax1.set_title("node groupings")
    # fig.colorbar(c1, ax = ax1)
    #
    # c2 = ax2.pcolormesh(pixel_nodes, cmap = 'jet')
    # ax2.invert_yaxis()
    # ax2.set_title("grouping after fill-in")
    # fig.colorbar(c2, ax = ax2)

    return (pixel_nodes, next_node_number - 1)


def label_pixels_by_node(preprocessed_img, num_colors, debug_print = False):
    puzzle = crop_to_puzzle(preprocessed_img)
    # print(puzzle.shape) # y by x by color
    # print(puzzle[0,0:5,:])

    # Use K-means to reduce number of colors
    pixel_colors = copy.deepcopy(puzzle)
    pixel_colors = np.reshape(pixel_colors, (puzzle.shape[0] * puzzle.shape[1], puzzle.shape[2]))
    pixel_colors = np.asarray(pixel_colors, dtype = np.float32)
    # print(pixel_colors.shape)
    # print(pixel_colors[0:5])

    # configure kmeans
    K = num_colors  # if you set K to equal the number of colors that are actually used in the puzzle, this should be very effective
    max_iters = 100
    epsilon = 1.0
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iters, epsilon)
    flags = cv2.KMEANS_RANDOM_CENTERS
    random_restarts = 10

    if debug_print:
        print(f"starting K means: max {max_iters} iterations, {random_restarts} random restarts")
    compactness, labels, centers = cv2.kmeans(pixel_colors, K + 1, None, termination_criteria, random_restarts, flags)

    palette_tolerance = 20
    # extract the color palette
    palette = crop_to_color_palette(preprocessed_img)
    new_centers = {}
    for center_index in range(len(centers)):
        center = centers[center_index]
        for e in range(num_colors):
            palette_entry = palette[10, ((palette.shape[1] // num_colors) * e) + 10]
            if abs(palette_entry[0] - center[0]) < palette_tolerance and abs(
                    palette_entry[1] - center[1]) < palette_tolerance and abs(
                palette_entry[2] - center[2]) < palette_tolerance:
                new_centers[center_index] = e

    ignore_labels = set(range(len(centers))) - set(new_centers)

    for entry in sorted(new_centers):
        print("COLOR {} means PALETTE {}".format(entry, new_centers[entry]))

    print("IGNORING COLORS {}".format(ignore_labels))

    if debug_print:
        print("K means complete!")
        # print("compactness =", compactness)
        print("centers =", centers)

    # compare the k-means colors to the color palette
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # centers_rgb = [(center[2] / 255.0, center[1] / 255.0, center[0] / 255.0) for center in centers]
    # bounds = np.arange(K + 1)
    # cmap = matplotlib.colors.ListedColormap(centers_rgb)
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    # cb2 = matplotlib.colorbar.ColorbarBase(ax1, cmap = cmap, norm = norm, orientation = 'horizontal')
    # ax1.set_title(f"k means colors (k = {K})")
    # #
    # ax2.imshow(cv2.cvtColor(palette, cv2.COLOR_BGR2RGB))
    # ax2.axis('off')
    # ax2.set_title("color palette from original image")
    plt.show()

    converted_puzzle = convert_to_kmeans_colors(labels, centers)
    # print(converted_puzzle.shape)
    # print(converted_puzzle[0,0:5,:])

    # compare the original puzzle to the k-means compressed version
    fig2, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(cv2.cvtColor(puzzle, cv2.COLOR_BGR2RGB))
    ax1.set_title("original image")
    ax1.axis('off')

    ax2.imshow(cv2.cvtColor(converted_puzzle, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"converted image ({K} colors)")
    ax2.axis('off')
    # plt.show()

    print("Parse contiguous regions")
    # parse contiguous regions (i.e. nodes) from converted colors
    num_labels = len(centers)
    flatten_labels = np.ravel(labels)
    pixel_labels = np.reshape(flatten_labels, (puzzle_height_y, puzzle_width_x))
    pixel_nodes, num_nodes = assign_pixels_to_nodes(pixel_labels, num_labels, ignore_labels)

    print("Count number of pixels of each color for each node")
    # count the number of pixels of each color for each node
    node_color_counts = {}
    for i in range(pixel_nodes.shape[0]):
        for j in range(pixel_nodes.shape[1]):
            # if pixel is still unlabeled, skip it
            if pixel_nodes[i, j] == 0:
                continue
            if pixel_nodes[i, j] not in node_color_counts:
                node_color_counts[pixel_nodes[i, j]] = collections.Counter()
            node_color_counts[pixel_nodes[i, j]][pixel_labels[i, j]] += 1

    print("Assign colors to each node")
    # assign colors to each node (assign to the most frequent node)
    node_colors = {}
    for node_num in node_color_counts:
        label = None
        max_count = float("-inf")
        for potential_label in node_color_counts[node_num]:
            if node_color_counts[node_num][potential_label] > max_count:
                label = potential_label
                max_count = node_color_counts[node_num][potential_label]
        # print(f"assigning node {node_num} to label {label}")
        if label in ignore_labels:
            label = None
        node_colors[node_num] = label

    return pixel_nodes, node_colors, num_nodes


# gets a list of neighboring pixel coordinates (y, x)
def get_neighbor_coords(pixel_y, pixel_x):
    neighbor_coords = []
    if pixel_y > 0:
        neighbor_coords.append((pixel_y - 1, pixel_x))
    if pixel_y < puzzle_height_y - 1:
        neighbor_coords.append((pixel_y + 1, pixel_x))
    if pixel_x > 0:
        neighbor_coords.append((pixel_y, pixel_x - 1))
    if pixel_x < puzzle_width_x - 1:
        neighbor_coords.append((pixel_y, pixel_x + 1))
    return neighbor_coords


def identify_adjacent_nodes(pixel_nodes, num_nodes, debug_print = False):
    puzzle_graph = {}
    counts = {}
    for i in range(1, num_nodes + 1):
        puzzle_graph[i] = set([])
        counts[i] = collections.Counter()

    for pixel_y in range(pixel_nodes.shape[0]):
        for pixel_x in range(pixel_nodes.shape[1]):
            neighbor_coords = get_neighbor_coords(pixel_y, pixel_x)

            this_label = pixel_nodes[pixel_y, pixel_x]

            if this_label > 0:
                for nbr_y, nbr_x in neighbor_coords:
                    if pixel_nodes[nbr_y, nbr_x] != this_label and pixel_nodes[nbr_y, nbr_x] > 0:
                        counts[this_label][pixel_nodes[nbr_y, nbr_x]] += 1
            else:
                # if this pixel was not labeled, it was neighboring to more than one contiguous region
                # so you can add edges between the nodes that are neighboring this pixel
                nbr_labels = set([])
                for nbr_y, nbr_x in neighbor_coords:
                    if pixel_nodes[nbr_y, nbr_x] > 0:
                        nbr_labels.add(pixel_nodes[nbr_y, nbr_x])
                for label1, label2 in itertools.combinations(nbr_labels, 2):
                    counts[label1][label2] += 1

    for this_node in range(1, num_nodes + 1):
        for potential_nbr in counts[this_node]:
            # set a reasonable threshold to avoid two opposite "corners" being marked as adjacent (corner cases, literally!)
            if counts[this_node][potential_nbr] > 15:
                puzzle_graph[this_node].add(potential_nbr)
                puzzle_graph[potential_nbr].add(this_node)
        if debug_print:
            print(f"neighbors of node {this_node} = {puzzle_graph[this_node]}")

    for this_node in range(1, num_nodes + 1):
        puzzle_graph[this_node] = frozenset(puzzle_graph[this_node])
    return puzzle_graph


def parse_image_graph(img_filename, num_colors, debug_print = False, debug_plots = False):
    img = get_original_image(img_filename)

    preprocessed_img = image_preprocessing(img)

    pixel_nodes, node_colors, num_nodes = label_pixels_by_node(preprocessed_img, num_colors,
                                                               debug_print = debug_print)
    if debug_print:
        print(f"assigned pixels to contiguous nodes! there are {num_nodes} nodes")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    c = ax1.pcolormesh(pixel_nodes, cmap = 'jet')
    ax1.invert_yaxis()
    ax1.set_title("node groupings")
    fig.colorbar(c, ax = ax1, spacing = 'uniform')

    puzzle = crop_to_puzzle(preprocessed_img)
    ax2.imshow(cv2.cvtColor(puzzle, cv2.COLOR_BGR2RGB))
    ax2.set_title("original image")
    if debug_plots:
        plt.show()

    # detect which nodes are adjacent to build the graph
    puzzle_graph = identify_adjacent_nodes(pixel_nodes, num_nodes, debug_print = debug_print)

    # dump to json for debugging
    # puzzle_graph_copy = {}
    # for i in range(1, num_nodes + 1):
    #     puzzle_graph_copy[i] = list(puzzle_graph[i])
    # with open("output_graph.json", 'w') as f:
    #     json.dump(puzzle_graph_copy, f)

    # print(node_colors)

    for node in list(node_colors):
        if node_colors[node] is None:
            del node_colors[node]

    return puzzle_graph, node_colors, pixel_nodes


def main():
    parser = argparse.ArgumentParser(
        description = 'Given a screenshot of a Kami 2 puzzle, construct the graph representation of the puzzle state.')
    parser.add_argument('img_filename', type = str,
                        help = 'path to the screenshot')
    parser.add_argument('num_colors', type = int,
                        help = 'number of colors in the puzzle (used to help label colors)')

    args = parser.parse_args()
    print(args.img_filename)

    parse_image_graph(args.img_filename, args.num_colors, debug_print = True, debug_plots = True)


if __name__ == "__main__":
    main()
