import datetime
import os
import xml.etree.ElementTree as ET

from .settings import XmlSettings


class XmlCreator:

    def __init__(self, settings: XmlSettings):
        self.settings: XmlSettings = settings

    def create_page_xml(self, regions, classification, image, output_dir, image_name):
        height = image.shape[0]
        width = image.shape[1]

        pcgts = ET.Element("PcGts", {
            "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15 "
                                  "http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15/pagecontent.xsd"
        })

        metadata = ET.SubElement(pcgts, "Metadata")
        creator = ET.SubElement(metadata, "Creator")
        creator.text = self.settings.author
        created = ET.SubElement(metadata, "Created")
        created.text = datetime.datetime.now().isoformat()
        last_change = ET.SubElement(metadata, "LastChange")
        last_change.text = datetime.datetime.now().isoformat()

        page = ET.SubElement(pcgts, "Page", {
            "imageFilename": image_name,
            "imageWidth": str(width),
            "imageHeight": str(height),
        })

        count = 0
        for region, prediction in zip(regions, classification):
            region_as_text = ET.SubElement(page, 'region', {"id": "r" + str(count), "type": str(prediction)})
            coords = ET.SubElement(region_as_text, "Coords",
                                   {"points": str(list(region.exterior.coords))})
            count += 1

        indent(pcgts)
        tree = ET.ElementTree(pcgts)
        tree.write(os.path.join(output_dir, self.settings.file_basename_prefix + image_name + '.' +
                                self.settings.file_basename_suffix), xml_declaration=True,
                   encoding=self.settings.encoding,
                   method="xml")


def indent(elem, level=0):
    i = os.linesep + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
