# FIXME: find a better way to refer to the bundled models that doesn't break when deployed as zip

import os
from importlib.util import find_spec

import ocr4all_segmentation.demo.models

default_classifier_model = os.path.join(list(ocr4all_segmentation.demo.models.__path__)[0], 'model')


if find_spec("pagecontent") is not None:
    import pagecontent.demo.model

    default_content_model = os.path.join(list(pagecontent.demo.model.__path__)[0], 'model')
else:
    default_content_model = None
