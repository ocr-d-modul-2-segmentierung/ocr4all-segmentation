from __future__ import absolute_import

import json
import os.path
import numpy as np

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    MetadataItemType,
    LabelsType, LabelType,
    TextRegionType,
    ImageRegionType,
    NoiseRegionType,
    CoordsType,
    to_xml,
)
from ocrd_utils import (
    getLogger,
    concat_padded,
    MIMETYPE_PAGE,
)
from pkg_resources import resource_string

from ocr4all_segmentation.segmentation.segmentation import Segmentator
from ocr4all_segmentation.segmentation.settings import SegmentationSettings
from .common import (
    image_from_page,
)

OCRD_TOOL = json.loads(resource_string(__name__, 'ocrd-tool.json').decode('utf8'))

TOOL = 'ocrd-ocr4all-segmentation'
LOG = getLogger('processor.OCR4AllSegmentation')
FALLBACK_IMAGE_GRP = 'OCR-D-SEG-BLOCK'


class OCR4AllSegmentation(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(OCR4AllSegmentation, self).__init__(*args, **kwargs)

    def process(self):
        """Performs segmentation on the input binary image

        Produces a PageXML file as output.
        """
        overwrite_regions = self.parameter['overwrite_regions']

        try:
            self.page_grp, self.image_grp = self.output_file_grp.split(',')
        except ValueError:
            self.page_grp = self.output_file_grp
            self.image_grp = FALLBACK_IMAGE_GRP
            LOG.info("No output file group for images specified, falling back to '%s'", FALLBACK_IMAGE_GRP)
        for n, input_file in enumerate(self.input_files):
            file_id = input_file.ID.replace(self.input_file_grp, self.image_grp)
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            metadata = pcgts.get_Metadata()  # ensured by from_file()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=self.ocrd_tool['steps'][0],
                                 value=TOOL,
                                 # FIXME: externalRef is invalid by pagecontent.xsd, but ocrd does not reflect this
                                 # what we want here is `externalModel="ocrd-tool" externalId="parameters"`
                                 Labels=[LabelsType(  # externalRef="parameters",
                                     Label=[LabelType(type_=name,
                                                      value=self.parameter[name])
                                            for name in self.parameter.keys()])]))
            page = pcgts.get_Page()
            if page.get_TextRegion():
                if overwrite_regions:
                    LOG.info('removing existing TextRegions')
                    page.set_TextRegion([])
                else:
                    LOG.warning('keeping existing TextRegions')

            page.set_AdvertRegion([])
            page.set_ChartRegion([])
            page.set_ChemRegion([])
            page.set_GraphicRegion([])
            page.set_ImageRegion([])
            page.set_LineDrawingRegion([])
            page.set_MathsRegion([])
            page.set_MusicRegion([])
            page.set_NoiseRegion([])
            page.set_SeparatorRegion([])
            page.set_TableRegion([])
            page.set_UnknownRegion([])

            page_image, page_xywh, _ = image_from_page(self.workspace, page, page_id)

            self._process_page(page, page_image, page_xywh, input_file.pageId, file_id)

            # Use input_file's basename for the new file -
            # this way the files retain the same basenames:
            file_id = input_file.ID.replace(self.input_file_grp, self.page_grp)
            if file_id == input_file.ID:
                file_id = concat_padded(self.page_grp, n)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.page_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.page_grp,
                                            file_id + '.xml'),
                content=to_xml(pcgts))

    def _process_page(self, page, page_image, page_xywh, pageId, file_id):
        settings = SegmentationSettings(debug=False, enable_preprocessing=False)
        # TODO: does this still need to be cropped or do we not need page_xywh?
        #       Same for points below
        #       page_image[page_xywh["x"]:page_xywh["w"], page_xywh["y"]:page_xywh["h"]]
        regions, classification = Segmentator(settings).segmentate_image(np.asarray(page_image))

        count = 0
        for region, prediction in zip(regions, classification):
            ID = "region%04d" % count
            points = str(list(region.exterior.coords))
            coords = CoordsType(points=points)
            # FIXME: these are not all types in the model, also check if they match
            if prediction == 1:
                page.add_TextRegion(TextRegionType(id=ID, Coords=coords))
            elif prediction == 2:
                page.add_ImageRegion(ImageRegionType(id=ID, Coords=coords))
            else:
                page.add_NoiseRegion(NoiseRegionType(id=ID, Coords=coords))
            count += 1
