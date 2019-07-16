from dataclasses import dataclass
from typing import List
import numpy as np
import copy


@dataclass
class Point:
    x: float
    y: float


class Line:
    def __init__(self, line: List[Point]):
        self.line: List[Point] = line

    def get_start_point(self):
        return self.line[0]

    def get_end_point(self):
        return self.line[-1]

    def get_average_line_height(self):
        return np.mean([point.y for point in self.line])

    def get_average_x_pos(self):
        return np.mean([point.x for point in self.line])

    def get_xy(self, x_offset=0, y_offset=0):
        x_list = []
        y_list = []
        for point in self.line:
            x_list.append(point.x + x_offset)
            y_list.append(point.y + y_offset)
        return x_list, y_list

    def get_line_length_y(self):
        return self.line[-1].y - self.line[0].y

    def __len__(self):
        return len(self.line)

    def __iter__(self):
        return iter(self.line)

    def __getitem__(self, key):
        return self.line[key]

    def __setitem__(self, key, value):
        self.line[key] = value

    def __str__(self):
        return "[{0}]".format(', '.join(map(str, self.line)))

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return self.line + other.line

    def __radd__(self, other):
        return other.line + self.line

    def l_append(self, value):
        self.line = value + self.line

    def r_append(self, value):
        self.line = self.line + value

    def scale_line(self, factor: float):
        self.line = [Point(point.x * factor, point.y * factor) for point in self.line]

    def __copy__(self):
        return Line(copy.copy(self.line))

    def __delitem__(self, key):
        del self.line[key]


class segment:
    def __init__(self, sub_image, path):
        self.sub_image = sub_image
        self.path = path

    def getbbox(self):
        rows = np.any(1 - self.sub_image * 1, axis=1)
        cols = np.any(1 - self.sub_image * 1, axis=0)
        y_val = np.where(rows)
        x_val = np.where(cols)
        yminmax = tuple(y_val[0][[0, -1]]) if y_val[0].size > 0 else (1, 1)
        xminmax = tuple(x_val[0][[0, -1]]) if x_val[0].size > 0 else (1, 1)
        ymin, ymax = yminmax[0], yminmax[1]
        xmin, xmax = xminmax[0], xminmax[1]
        return (ymin, ymax + 1), (xmin, xmax + 1)


@dataclass
class ImageData:
    path: str = None
    height: int = None
    image: np.array = None
    average_letter_height: int = None
    binary_image: np.array = None
    pixel_classifier_prediction: np.array = None