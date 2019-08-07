from enum import IntEnum
from typing import NamedTuple, Optional


class SegmentationSettings(NamedTuple):
    debug: bool = False
    debug_page_content_mask = False

    anchor_points_distance: int = 100  # every xth pixel a point is generated
                                       # for the line which cuts the image is segments

    line_window: int = 5  # window of the line points to best fit

    # preprocessing parameters
    enable_preprocessing: bool = False
    min_size_objects: int = 5
    min_image_height: int = 100
    min_cc_factor_size: float = 5.0
    min_window_width: int = 3.0
    remove_big_contours: bool = True
    big_contour_height_ratio = 10

    # page content model settings
    page_content_model: Optional[str] = None  # path to page content model
    page_content_debug: bool = False
    page_content_model_debug: bool = False

    # postprocessing parameter
    validate_vertical_lines: bool = True
    validate_vertical_lines_width: int = 3
    validate_vertical_line_blakness: float = 0.995

    resize_regions_to_original_size: bool = True

    fit_line = True


class RegionClassifierSettings(NamedTuple):
    debug: bool = False
    model: Optional[str] = None
    target_line_space_height: int = 6
    line_space_height: int = 0
    processes: int = 8
    write_image: bool = False
