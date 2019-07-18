from ocr4all_segmentation.segmentation.settings import  RegionClassifierSettings
from matplotlib import pyplot as plt
from definitions import default_classifier_model
import os
from pagesegmentation.lib.predictor import PredictSettings
from ocr4all_segmentation.pixelclassifier.predictor import PCPredictor
import numpy as np
from ocr4all_segmentation.segmentation.util import compute_char_height
from ocr4all_segmentation.segmentation.datatypes import ImageData
from functools import partial
import multiprocessing
import tqdm
import cv2


class RegionClassifier:
    def __init__(self, settings: RegionClassifierSettings):
        self.settings = settings
        self.page_predictor = None
        model_path = self.settings.model if self.settings.model else default_classifier_model
        self.predictor = None
        if model_path:
            pcsettings = PredictSettings(
                mode='meta',
                network=os.path.abspath(model_path),
                output=None,
                high_res_output=False
            )
            self.predictor = PCPredictor(pcsettings, settings.target_line_space_height)

    def classify(self, images, regions):
        create_data_partial = partial(create_data, avg_letter_height=self.settings.line_space_height)
        if len(images) <= 1:
            data = list(map(create_data_partial, images))
        else:
            with multiprocessing.Pool(processes=self.settings.processes) as p:
                data = [v for v in tqdm.tqdm(p.imap(create_data_partial, images), total=len(images))]
        for i, prob in enumerate(self.predictor.predict(data)):
            data[i].pixel_classifier_prediction = prob
            rounded = np.around(prob)

            debug_image = np.zeros(data[i].image.shape)
            classification = []
            for x in regions:
                path = list(x.exterior.coords)
                mask = generate_content_mask(path, data[i].image.shape)
                max_class = vote_region_class(rounded, mask)
                classification.append(max_class)
                if self.settings.debug:
                    debug_image = debug_image + mask * max_class

            if self.settings.debug:
                f, ax = plt.subplots(1, 3, True, True)
                ax[0].imshow(data[i].image)
                ax[1].imshow(rounded)
                ax[2].imshow(debug_image)
                plt.show()
        return classification


def vote_region_class(pred: np.ndarray, mask) -> np.ndarray:
    prebin = (pred * mask).flatten().astype(np.uint8)
    bins = np.bincount(prebin)
    maxclass = 0 if bins.size <= 1 else np.argmax(bins[1:]) + 1
    return maxclass


def create_data(image: np.ndarray, avg_letter_height: int) -> ImageData:
    binary_image = image.astype(np.uint8) / 255
    if avg_letter_height == 0:
        avg_letter_height = compute_char_height(image)
        print(avg_letter_height)
    image_data = ImageData(image=binary_image, average_letter_height=avg_letter_height,
                           binary_image=binary_image)
    return image_data


def generate_content_mask(path, image_shape):
    mask = np.zeros(image_shape)
    cv2.fillPoly(mask, [np.int32(path)], 255)
    return mask / 255

