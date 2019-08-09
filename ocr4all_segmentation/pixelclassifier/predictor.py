from typing import Generator, List

import numpy as np
from pagesegmentation.lib.dataset import DatasetLoader, SingleData
from pagesegmentation.lib.predictor import Predictor, PredictSettings
from skimage.transform import resize

from ocr4all_segmentation.segmentation.datatypes import ImageData


class PCPredictor:
    def __init__(self, settings: PredictSettings, height=20):
        self.height = height
        self.settings = settings
        self.predictor = Predictor(settings)

    def predict(self, images: List[ImageData]) -> Generator[np.array, None, None]:
        dataset_loader = DatasetLoader(self.height, prediction=True)
        data = dataset_loader.load_data(
            [SingleData(binary=i.image * 255, image=i.image * 255, line_height_px=i.average_letter_height) for i in images]
        )
        for i, pred in enumerate(self.predictor.predict(data)):
            prob = pred.labels
            pred = resize(prob, pred[2].original_shape, preserve_range=True)
            yield pred
