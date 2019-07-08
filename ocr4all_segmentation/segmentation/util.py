# Important Imports
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
import cv2
from itertools import tee
from ocr4all_segmentation.segmentation.datatypes import Point, Line
from scipy.interpolate import interpolate
from matplotlib import pyplot as plt
import cv2
import numpy as np


# image = PIL.Image, n = Number of Segments
# ignoreBottomTop = Segmentation of top and bottom of Image
# axis = 0 (for vertical-lines) or 1 (for horizontal-lines)
# Returns a gray image, PIL Image.
def recursive_xy_cut(image_arr, plateau_size = 3, ignoreBottomTop = False, axis = 1):
    # Sum the pixels along given
    image_arr_inv = image_arr
    sum_vals = image_arr_inv.sum(axis = axis)
    # Get the indices of the peaks

    peaks, _ = find_peaks(sum_vals, plateau_size=plateau_size, height= 0.95 * image_arr.shape[axis])
    # Temp variable to create segment lines i.e. 0 out the required values.
    temp = np.ones(image_arr.shape)
    # Skip top and bottom segmentation or not (depends on the param)
    #for peak in peaks[1:-1 if ignoreBottomTop else ]:
    for peak in peaks[1:-1] if ignoreBottomTop else peaks:
        if axis == 1:
            temp[range(peak-1, peak+1)] = 0
        else:
            temp[:, range(peak-2, peak+2)] = 0
    #plt.imshow(np.uint8(image_arr * temp))
    #plt.show()
    return peaks


def iteration(image: np.ndarray, value: int) -> np.ndarray:
    """
    This method iterates over the provided image by converting 255's to 0's if the number of consecutive 255's are
    less the "value" provided
    """

    rows, cols = image.shape
    for row in range(0, rows):
        try:
            start = image[row].tolist().index(0)  # to start the conversion from the 0 pixel
        except ValueError:
            start = 0  # if '0' is not present in that row

        count = start
        for col in range(start, cols):
            if image[row, col] == 0:
                if (col - count) <= value and (col - count) > 0:
                    image[row, count:col] = 0
                count = col
    return image


def rlsa(image: np.ndarray, horizontal: bool = True, vertical: bool = True, value: int = 0) -> np.ndarray:
    """
    rlsa(RUN LENGTH SMOOTHING ALGORITHM) is to extract the block-of-text or the Region-of-interest(ROI) from the
    document binary Image provided. Must pass binary image of ndarray type.
    """

    if isinstance(image, np.ndarray):  # image must be binary of ndarray type
        value = int(value) if value >= 0 else 0  # consecutive pixel position checker value to convert 255 to 0
        # RUN LENGTH SMOOTHING ALGORITHM working horizontally on the image
        if horizontal:
            image = iteration(image, value)

            # RUN LENGTH SMOOTHING ALGORITHM working vertically on the image
        if vertical:
            image = image.T
            image = iteration(image, value)
            image = image.T
    else:
        print('Image must be an numpy ndarray and must be in binary')
        image = None
    return image


def compute_avg_cc_height(image: np.array):
    # labeled, nr_objects = ndimage.label(img > 128)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((np.invert(image)*255).astype(np.uint8), 4)
    heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, len(stats))]
    return np.mean(heights)


def compute_char_height(image: np.array):

    # labeled, nr_objects = ndimage.label(img > 128)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 4)

    possible_letter = [False] + [0.5 < (stats[i, cv2.CC_STAT_WIDTH] / stats[i, cv2.CC_STAT_HEIGHT]) < 2
                                 and 10 < stats[i, cv2.CC_STAT_HEIGHT] < 60
                                 and 5 < stats[i, cv2.CC_STAT_WIDTH] < 50
                                 for i in range(1, len(stats))]

    valid_letter_heights = stats[possible_letter, cv2.CC_STAT_HEIGHT]

    valid_letter_heights.sort()
    try:
        mode = valid_letter_heights[int(len(valid_letter_heights) / 2)]
        return mode
    except IndexError:
        return None


def compute_line_space_height(image: np.array):
    counts = np.sum(image == 0, axis=1)
    peaks = find_peaks(counts)
    peak_difference = []
    for x_ind, x in enumerate(peaks[0]):
        if x_ind + 1 < len(peaks[0]):
            peak_difference.append(peaks[0][x_ind + 1] - peaks[0][x_ind])
    peak_difference.sort()
    return np.median(peak_difference)


def vertical_runs(img: np.array) -> [int, int]:
    img = np.transpose(img)
    h = img.shape[0]
    w = img.shape[1]
    transitions = np.transpose(np.nonzero(np.diff(img)))
    white_runs = [0] * (w + 1)
    black_runs = [0] * (w + 1)
    a, b = tee(transitions)
    next(b, [])
    for f, g in zip(a, b):
        if f[0] != g[0]:
            continue
        tlen = g[1] - f[1]
        if img[f[0], f[1] + 1] == 1:
            white_runs[tlen] += 1
        else:
            black_runs[tlen] += 1

    for y in range(h):
        x = 1
        col = img[y, 0]
        while x < w and img[y, x] == col:
            x += 1
        if col == 1:
            white_runs[x] += 1
        else:
            black_runs[x] += 1

        x = w - 2
        col = img[y, w - 1]
        while x >= 0 and img[y, x] == col:
            x -= 1
        if col == 1:
            white_runs[w - 1 - x] += 1
        else:
            black_runs[w - 1 - x] += 1

    black_r = np.argmax(black_runs) + 1
    # on pages with a lot of text the staffspaceheigth can be falsified.
    # --> skip the first elements of the array, we expect the staff lines distance to be at least twice the line height
    white_r = np.argmax(white_runs[black_r * 3:]) + 1 + black_r * 3
    return white_r, black_r


def best_line_fit(img: np.array, line: Line, line_thickness: int = 3, max_iterations: int = 30,
                  scale: float = 1.0, skip_startend_points: bool = False) -> Line:
    current_blackness = get_blackness_of_vertical_line(line, img)
    best_line = line.__copy__()

    change = True
    iterations = 0

    while change:
        if iterations > max_iterations:
            break
        change = False
        for point_ind, point in enumerate(best_line):
            if skip_startend_points:
                if point_ind == 0 or point_ind == len(best_line):
                    continue
            y, x = point.y, point.x
            scaled_line_thickness = line_thickness * np.ceil(scale).astype(int)
            for i in range(1, scaled_line_thickness + 1):
                if x + i < line[point_ind].x + scaled_line_thickness and x + i < img.shape[1] - 1:
                    test_line = best_line.__copy__()
                    test_line[point_ind] = Point(x + i, y)
                    blackness = get_blackness_of_vertical_line(test_line, img)

                    if blackness < current_blackness:
                        change = True
                        current_blackness = blackness
                        best_line[point_ind] = Point(x + i, y)
                if x - i > line[point_ind].x - scaled_line_thickness and x - i > 1:
                    test_line = best_line.__copy__()
                    test_line[point_ind] = Point(x - i, y)
                    blackness = get_blackness_of_vertical_line(test_line, img)

                    if blackness < current_blackness:
                        change = True
                        current_blackness = blackness
                        best_line[point_ind] = Point(x - i, y)

        iterations += 1
    return best_line


def get_blackness_of_vertical_line(line: Line, image: np.ndarray, window=1, vote=False) -> int:

    image = image * 1.0
    x_list, y_list = line.get_xy()
    func = interpolate.interp1d(y_list, x_list)
    y_start, y_end = int(y_list[0]), int(y_list[-1])
    y_list_new = np.arange(y_start, y_end-1)
    x_new = func(y_list_new)
    x_new[x_new > image.shape[1] - 1] = image.shape[1] - 1
    x_new_int = np.floor(x_new + 0.5).astype(int)

    index_y = np.empty(0, dtype=np.uint8)
    index_x = np.empty(0, dtype=np.uint8)

    window_cp = 0
    while window_cp != window:

        if window_cp == 0:
            index_y = np.concatenate((index_y, y_list_new))
            index_x = np.concatenate((index_x, x_new_int))
        else:
            if np.max(index_x + window_cp) < image.shape[1]:
                index_y = np.concatenate((index_y, y_list_new))
                index_x = np.concatenate((index_x, x_new_int + window_cp))
            if np.min(index_x - window_cp) >= 0:
                index_y = np.concatenate((index_y, y_list_new))
                index_x = np.concatenate((index_x, x_new_int - window_cp))
            if vote is True:
                plt.imshow(image)
                plt.plot(x_new_int - window_cp, y_list_new)
                plt.show()
        window_cp = window_cp + 1

    blackness = 0
    if vote and window > 0:
        for x in range(0, max(window + 1, 5)):
            if np.max(index_x + x) < image.shape[1]:
                indexes = (index_y, index_x + x)
                blackness = np.mean(image[indexes])

                if blackness == 1:
                    return blackness
            if np.min(index_x - x) >= 0:
                indexes = (index_y, index_x - x)
                blackness = np.mean(image[indexes])

                if blackness == 1:
                    return blackness

    else:
        indexes = (index_y, index_x)
        blackness = np.mean(image[indexes])
    return blackness


def generate_vertical_line(x_value, image, anchor_points=100):
    height, width = image.shape
    anchor_points = max(2, np.ceil(height / anchor_points))
    spaced_numbers = np.linspace(1, height - 1, anchor_points, endpoint=True)
    line = Line([Point(x=x_value, y=int(sp_n)) for sp_n in spaced_numbers])
    return line


def generate_horizontal_line(y_value, image, anchor_points=35):
    height, width = image.shape
    anchor_points = max(np.ceil(width / anchor_points), 2)

    spaced_numbers = np.linspace(1, width - 1, anchor_points, endpoint=True)
    line = Line([Point(x=(sp_n), y=y_value) for sp_n in spaced_numbers])
    return line


def prune_neighbor_lines(lines):
    change = True
    while change:
        change = False
        for x in range(len(lines) - 1):
            for y in range(x + 1, len(lines)):
                if check_if_poly_line_intersects(lines[x], lines[y]):
                    change = True
                    del lines[y]
                    break
            else:
                continue
            break


def check_if_poly_line_intersects(line1: Line, line2: Line):
    for segment in range(len(line1) - 1):
        point1s1 = line1[segment]
        point2s1 = line1[segment + 1]
        for ind2 in range(len(line2) - 1):
            point1s2 = line2[ind2]
            point2s2 = line2[ind2 + 1]

            if do_intersect((point1s1.x, point1s1.y), (point2s1.x, point2s1.y), (point1s2.x, point1s2.y), (point2s2.x, point2s2.y)):
                return True


def on_segment(p, q, r):
    '''Given three colinear points p, q, r, the function checks if
    point q lies on line segment "pr"
    '''
    if (max(p[0], r[0]) >= q[0] >= min(p[0], r[0]) and
            max(p[1], r[1]) >= q[1] >= min(p[1], r[1])):
        return True
    return False


def orientation(p, q, r):
    '''Find orientation of ordered triplet (p, q, r).
    The function returns following values
    0 --> p, q and r are colinear
    1 --> Clockwise
    2 --> Counterclockwise
    '''

    val = ((q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]))
    if val == 0:
        return 0  # colinear
    elif val > 0:
        return 1   # clockwise
    else:
        return 2  # counter-clockwise


def do_intersect(p1, q1, p2, q2):
    '''Main function to check whether the closed line segments p1 - q1 and p2
       - q2 intersect'''
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    # p1, q1 and p2 are colinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False # Doesn't fall in any of the above cases


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def validate_line(cuts, image, threhsold = 1.0):
    y_cuts = []
    for y_val in cuts:
        line = generate_horizontal_line(y_val, image)
        blackness = get_blackness_of_line(line, image)
        if blackness >= threhsold:
            y_cuts.append(y_val)
    return y_cuts


def get_blackness_of_line(line: Line, image: np.ndarray) -> int:
    x_list, y_list = line.get_xy()
    func = interpolate.interp1d(x_list, y_list)
    x_start, x_end = int(x_list[0]), int(x_list[-1])
    x_list_new = np.arange(x_start, x_end-1)
    y_new = func(x_list_new)
    y_new[y_new > image.shape[0] - 1] = image.shape[0] - 1
    y_new_int = np.floor(y_new + 0.5).astype(int)
    indexes = (np.array(y_new_int), np.array(x_list_new))

    blackness = np.mean(image[indexes])
    return blackness
