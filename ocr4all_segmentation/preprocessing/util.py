from skimage.transform import rotate, resize
from typing import NamedTuple
import numpy as np
from skimage.morphology import remove_small_holes
from shapely.geometry import Polygon


class ImageRescaler():

    def __init__(self, max_width=800, threshold=0.1):
        self.max_width = max_width
        self.threshold = threshold

    def rescale(self, binary_image: np.array):
        o_height, o_width = binary_image.shape
        factor = o_width / self.max_width
        height = int(o_height / factor)

        image_resized = resize(binary_image / 255, (height, self.max_width), preserve_range=True, anti_aliasing=False,
                               order=0)

        return image_resized, factor

    def remove_small_objects(self, binary_image: np.array, min_size=10):
        return remove_small_holes(binary_image, min_size=min_size)


class PolygonRescaler(Polygon):

    @property
    def xy(self):
        pass

    def _set_coords(self, ob):
        pass

    def _get_coords(self):
        pass

    @property
    def coords(self):
        pass

    @property
    def __array_interface__(self):
        pass

    def __init__(self):
        super().__init__()

