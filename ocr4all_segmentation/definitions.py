import os, pagecontent.demo.model, ocr4all_segmentation.demo.models

#FIXME: find a better way to refer to the bundled models that doesn't break when deployed as zip
default_content_model = os.path.join(list(pagecontent.demo.model.__path__)[0], 'model')
default_classifier_model = os.path.join(list(ocr4all_segmentation.demo.models.__path__)[0], 'model')
