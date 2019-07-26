from ocr4all_segmentation.segmentation.util import recursive_xy_cut, rlsa, compute_char_height,\
    compute_line_space_height, vertical_runs, generate_vertical_line, best_line_fit, prune_neighbor_lines, \
    compute_avg_cc_height, validate_line, get_blackness_of_vertical_line
import numpy as np
from typing import List
from scipy.signal import find_peaks
from ocr4all_segmentation.segmentation.settings import SegmentationSettings
from ocr4all_segmentation.segmentation.datatypes import Point, Line
from ocr4all_segmentation.preprocessing.util import ImageRescaler
from PIL import Image
from subprojects.page_content.pagecontent.detection.detection import PageContentDetection
from subprojects.page_content.pagecontent.detection.settings import PageContentSettings
import cv2
from skimage.morphology import remove_small_objects
from ocr4all_segmentation.segmentation.datatypes import segment
from ocr4all_segmentation.segmentation.util import pairwise
from definitions import default_content_model
from matplotlib.path import Path
from shapely import geometry
from shapely.affinity import scale
from shapely.geometry import Polygon
from skimage.draw import polygon
from ocr4all_segmentation.segmentation.post_processing import RegionClassifier, RegionClassifierSettings


class Segmentator:
    def __init__(self, settings: SegmentationSettings):
        self.settings = settings
        self.page_predictor = None
        model_path = self.settings.page_content_model if self.settings.page_content_model else default_content_model
        if model_path:
            settings_border_predictor = PageContentSettings(
                model=model_path,
                debug=self.settings.page_content_debug,
                debug_model=self.settings.page_content_model_debug,
            )
            self.page_predictor = PageContentDetection(settings_border_predictor)
            self.r_settings = RegionClassifierSettings()
            self.r_classifier = RegionClassifier(self.r_settings)

    def segmentate_image_path(self, path):
        _image = np.array(Image.open(path))
        content_polygon = list(self.page_predictor.detect([_image]))[0]
        if content_polygon is not None:
            mask = draw_polygons([content_polygon], _image.shape)
            _image[mask < 1] = 255
        _rescaled_image, rescale_factor = ImageRescaler().rescale(_image)
        _np_image = _rescaled_image.astype(np.bool)
        avg_cc_height, median_cc_height = compute_avg_cc_height(_np_image)
        if self.settings.remove_big_contours:
            _rescaled_image = preprocess(_rescaled_image, avg_cc_height, self.settings.big_contour_height_ratio)
        inverted = (_rescaled_image == 0)
        cleansed_image = np.invert(remove_small_objects(inverted,
                                                        min_size=self.settings.min_size_objects, in_place=True))

        if self.settings.debug_page_content_mask:
            f, ax = plt.subplots(1, 2, True, True)
            ax[0].imshow(mask)
            ax[1].imshow(_image)
            plt.show()

        y_cuts = list(recursive_xy_cut(cleansed_image))
        y_cuts.insert(0, 0)
        y_cuts.append(_rescaled_image.shape[0])
        y_cuts = validate_line(y_cuts, cleansed_image)

        def generate_sub_images(image, cuts, axis=0) -> List[segment]:
            _sub_images = []
            previous = 0
            for x in range(1, len(cuts)):
                if axis is 0:
                    subarray = image[previous:cuts[x]]
                    _sub_images.append(segment(subarray, (previous, x, 0, image.shape[1])))

                else:
                    subarray = image[:, previous:cuts[x]]
                    _sub_images.append(segment(subarray, (0, image.shape[0], previous, x,)))

                previous = cuts[x]
            return _sub_images
        sub_images_horizontal = generate_sub_images(cleansed_image, y_cuts)

        regions = []

        for image in sub_images_horizontal:
            bbox = image.getbbox()
            avg_cc_height, median_cc_height = compute_avg_cc_height(image.sub_image)
            bbox_height = bbox[0][1] - bbox[0][0]
            _x = get_bp_distribution(image.sub_image[bbox[0][0]:bbox[0][1], bbox[1][0]: bbox[1][1]], 0)
            x_cuts = self.find_cut_points(_x, image.sub_image[bbox[0][0]:bbox[0][1], bbox[1][0]: bbox[1][1]])
            if self.settings.validate_vertical_lines and x_cuts :
                new_x_cuts = []
                for x_ind, x in enumerate(x_cuts):
                    if x_ind == 0 or x_ind + 1 == len(x_cuts):
                        new_x_cuts.append(x)
                        continue
                    line_length = x.get_line_length_y()
                    if line_length < 500:
                        if 0.5 < bbox_height / avg_cc_height <= 5:
                            blackness = get_blackness_of_vertical_line(x, image.sub_image[bbox[0][0]:bbox[0][1],
                                                                          bbox[1][0]: bbox[1][1]],
                                                                       self.settings.validate_vertical_lines_width
                                                                       * 10)
                        elif 5 < bbox_height / avg_cc_height < 25:
                            blackness = get_blackness_of_vertical_line(x, image.sub_image[bbox[0][0]:bbox[0][1],
                                                                            bbox[1][0]: bbox[1][1]],
                                                                       int(median_cc_height / 4),
                                                                       vote=True, debug=False)

                        else:
                            blackness = get_blackness_of_vertical_line(x, image.sub_image[bbox[0][0]:bbox[0][1],
                                                                            bbox[1][0]: bbox[1][1]],
                                                                       self.settings.validate_vertical_lines_width - 1,
                                                                       vote=True, debug=False)

                        if blackness >= self.settings.validate_vertical_line_blakness:
                            new_x_cuts.append(x)
                    else:
                        blackness = get_blackness_of_vertical_line(x, image.sub_image[bbox[0][0]:bbox[0][1],
                                                                        bbox[1][0]: bbox[1][1]],
                                                                   0, vote=True, debug=False)
                        if blackness >= self.settings.validate_vertical_line_blakness:
                            new_x_cuts.append(x)

                x_cuts = new_x_cuts
            if x_cuts:
                for l1, l2 in pairwise(x_cuts):
                    line1 = l1
                    line2 = l2
                    listline1 = list(zip(*line1.get_xy(x_offset=bbox[1][0], y_offset=bbox[0][0] + image.path[0])))
                    listline2 = list(zip(*line2.get_xy(x_offset=bbox[1][0], y_offset=bbox[0][0] + image.path[0])))
                    path = listline1 + listline2[::-1]

                    cleansed_image_cp = cleansed_image.copy()
                    mask = generate_content_mask(path, cleansed_image.shape)
                    cleansed_image_cp[mask < 1] = 255
                    _segment = segment(cleansed_image_cp, path)

                    (y1b, y2b), (x1b, x2b) = _segment.getbbox()
                    xmin, xmax = min(np.asarray(listline1)[:, 0]), max(np.asarray(listline2)[:, 0])
                    ymin, ymax = min(np.asarray(listline2)[:, 1]), max(np.asarray(listline2)[:, 1])

                    list1x = list(np.asarray(listline1)[:, 0] + (x1b - xmin))
                    list1y = list(np.asarray(listline1)[:, 1])
                    list2x = list(np.asarray(listline2)[:, 0] + (x2b - xmax))
                    list2y = list(np.asarray(listline2)[:, 1])
                    path = list(zip(list1x, list1y)) + list(zip(list2x, list2y))[::-1]
                    regions.append(geometry.Polygon(path))
            else:
                regions.append(geometry.Polygon([[bbox[1][0], bbox[0][0] + image.path[0]],
                                                 [bbox[1][1], bbox[0][0] + image.path[0]],
                                                 [bbox[1][1], bbox[0][1] + image.path[0]],
                                                 [bbox[1][0], bbox[0][1] + image.path[0]]]))

        if self.settings.resize_regions_to_original_size:
            regions = self.resize_to_original(regions, rescale_factor)
            if self.settings.debug:
                plt.imshow(_image)
                for polygon in regions:
                    x, y = polygon.exterior.xy
                    plt.plot(x, y)
                plt.show()
        else:
            if self.settings.debug:
                plt.imshow(cleansed_image)
                for polygon in regions:
                    x, y = polygon.exterior.xy
                    plt.plot(x, y)
                plt.show()

        classification = self.r_classifier.classify([_image], regions, [image_path])
        return regions, classification

    def resize_to_original(self, regions, factor):
        new_regions = [scale(region, factor, factor, origin=(0, 0, 0)) for region in regions]
        return new_regions

    def find_cut_points(self, count: List[int], image, addstartend=True):
        lines = []

        image_height, width = image.shape
        indexes = np.where(count[:-1] != count[1:])
        base_height = get_base_height(count)

        multiplikator = 0.95 if base_height / image_height < 0.85 else 0.99
        peaks, keys = find_peaks(count, height=max(image_height - abs(base_height - multiplikator * image_height),
                                                   image_height * multiplikator))

        if peaks.size != 0:
            peaks = np.concatenate(([indexes[0][0]], peaks, [indexes[0][-1]]))
        else:
            return
        bases = get_left_right_base(count, peaks)
        peak1, bases1 = get_bases(count, peaks, bases)
        for ind_base, base in enumerate(bases1):
            xl, xr = base
            vertical_line_ypos = xl if count[xl] > count[xr] else xr
            vertical_line_ypos = vertical_line_ypos if count[vertical_line_ypos] > count[int(np.mean(base))] else int(
                np.mean(base))
            new_line = generate_vertical_line(vertical_line_ypos,
                                              image=image, anchor_points=self.settings.anchor_points_distance)
            if self.settings.fit_line:
                new_line = best_line_fit(1 - (image / 255), new_line, line_thickness=self.settings.line_window)

            lines.append(new_line)
        if addstartend:
            lines.insert(0, Line([Point(x=0, y=0), Point(x=0, y=image.shape[0])]))
            lines.append(Line([Point(x=image.shape[1], y=0), Point(x=image.shape[1], y=image.shape[0])]))
        prune_neighbor_lines(lines)
        return lines


def generate_content_mask(path, image_shape):
    mask = np.zeros(image_shape)
    cv2.fillPoly(mask, [np.int32(path)], 255)
    return mask / 255


def calculate_sub_image(path, image):
    print('Next iteration')
    image_cp = image.copy()
    mbpath = Path(np.asarray(path))
    extents = mbpath.get_extents()
    image_slice = image_cp[int(extents.ymin):int(extents.ymax), int(extents.xmin): int(extents.xmax)]
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(image_slice)
    ax[1].imshow(image_cp)
    plt.show()

    return image_cp[int(extents.ymin):int(extents.ymax), int(extents.xmin): int(extents.xmax)]


def get_bp_distribution(image: np.array, axis=0):
    counts = np.sum((1 - image) == 0, axis=axis)
    return counts


def get_base_height(count: List[int]):
    indexes = np.where(count[:-1] != count[1:])
    base_line = np.average(np.array(count)[indexes])
    return base_line


def get_left_right_base(list_t, peaks_ind):
    new_peaks_bases = []
    for x in peaks_ind:
        new_peaks_bases.append((get_local_minima(list(list_t), x, step=-1),
                               get_local_minima(list(list_t), x)))

    return new_peaks_bases


def get_bases(list_t, ind, ind_bases):
    new_indices = []
    new_bases = []
    for peak_ind, peak in enumerate(ind):
        peak_value = list_t[peak]
        h_value_left = list_t[ind_bases[peak_ind][0]]
        h_value_right = list_t[ind_bases[peak_ind][1]]
        thresh = (abs(peak_value - h_value_left) + abs(peak_value - h_value_right)) / peak_value
        if thresh > 0.1:
            new_indices.append(peak)
            new_bases.append(ind_bases[peak_ind])
    return new_indices, new_bases


def get_local_minima(list, peak_ind, threshold = 0.3, step=1):
    global_height_ind = peak_ind
    global_height_value = list[peak_ind]
    local_height_ind = global_height_ind
    local_height_value = global_height_value
    global_minimum_ind = global_height_ind
    global_minimum_value = global_height_value
    length = len(list) if step > 0 else 0
    for x in range(peak_ind + 1 * step, length, step):
        value = list[x]
        if value < global_minimum_value:
            global_minimum_ind = x
            global_minimum_value = value
        elif value == global_minimum_value:
            pass
        elif value > global_height_value:
            return global_minimum_ind
        elif value < local_height_value - (local_height_value - global_minimum_value) * threshold:
            local_height_ind = x
            local_height_value = list[x]
        else:
            return global_minimum_ind
    return global_minimum_ind


def preprocess(image, av_cc_height, max_contur_ratio, debug=False):
    contours, hierarchy = cv2.findContours(image.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(image.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    reverse = largest_areas[::-1]
    for contour in reverse[1:]:
        x, y, w, h = cv2.boundingRect(contour)
        if h >= max_contur_ratio * av_cc_height:
            cv2.drawContours(mask, [contour], 0, (255, 255, 255, 255), -1)
        else:
            break
    image[mask == 255] = 1
    if debug:
        plt.imshow(image)
        plt.show()
    return image


def draw_polygons(_polygons: List[Polygon], image_shape):
    mask = np.zeros(image_shape)
    for _poly in _polygons:
        x, y = _poly.exterior.xy
        rr, cc = polygon(y, x)
        mask[cc, rr] = 1
    return mask


if __name__ == '__main__':
    import pickle
    import os
    from matplotlib import pyplot as plt
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    page_path = os.path.join(project_dir, 'ocr4all_segmentation/demo/104.png')
    _page_content_model = os.path.join(project_dir, 'subprojects/page_content/pagecontent/demo/model/model')
    _settings = SegmentationSettings(debug=True)
    _segmentator = Segmentator(_settings)
    _segmentator.segmentate_image_path(page_path)

