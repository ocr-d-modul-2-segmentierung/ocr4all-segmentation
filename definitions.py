import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root

default_content_model = os.path.join(ROOT_DIR, 'subprojects/page_content/pagecontent/demo/model/model')

default_classifier_model = os.path.join(ROOT_DIR, 'ocr4all_segmentation/demo/models/model')
