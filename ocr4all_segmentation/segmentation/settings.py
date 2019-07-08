from enum import IntEnum
from typing import NamedTuple, Optional


class SegmentationSettings(NamedTuple):
    debug: bool = False
    debug_page_content_mask = False

    anchor_points_distance: int = 100  # every xth pixel a point is generated
                                       # for the line which cuts the image is segments

    line_window: int = 5  # window of the line points to best fit

    # advanced parameters
    min_size_objects: int = 5
    min_image_height: int = 100
    min_cc_factor_size: float = 5.0
    min_window_width: int = 3.0

    page_content_model: Optional[str] = None  # path to page content model
    page_content_debug: bool = False
    page_content_model_debug: bool = False

    # not yet implemented
    max_contour_size_ratio: float = 20   # cc with area of x times avg_cc_area are removed

    # not yet implemented
    validate_vertical_lines: bool = True
    validate_vertical_lines_width: int = 5
