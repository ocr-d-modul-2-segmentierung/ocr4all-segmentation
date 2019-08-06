import argparse
import glob
from ocr4all_segmentation.segmentation.segmentation import Segmentator
from ocr4all_segmentation.segmentation.settings import SegmentationSettings
import tqdm
from ocr4all_segmentation.xml.xml_creator import XmlCreator, XmlSettings
import os
import numpy as np


def glob_all(filenames):
    files = []
    for f in filenames:
        files += glob.glob(f)

    return files


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', ''):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='Detects page segments in historical documents')
    parser.add_argument("--binary", "--image", type=str, required=True, nargs="+",
                        help="directory name of the binary images")
    parser.add_argument("--processes", type=int, default=8,
                        help="Number of processes to use")
    parser.add_argument("--xml", type=str2bool, default=False,
                        help="Create Xml files")
    parser.add_argument("--output", type=str, required=False, default='.',
                        help="output directory for xml files")
    parser.add_argument("--debug", type=str2bool, default=False,
                        help="Display debug images")
    parser.add_argument("--preprocess", type=str2bool, default=False,
                        help="Use preprocessing to remove page border")
    args = parser.parse_args()

    bin_file_paths = sorted(glob_all(args.binary))
    print("Loading {} files".format(len(bin_file_paths)))

    settings = SegmentationSettings(
        debug=args.debug,
        enable_preprocessing=args.preprocess,
    )

    xml_settings = XmlSettings()
    xml_creator = XmlCreator(xml_settings)
    segmentator = Segmentator(settings)
    for i in tqdm.tqdm(bin_file_paths, total=len(bin_file_paths)):
        regions, classification = segmentator.segmentate_image_path(i)
        if args.xml:
            xml_creator.create_page_xml(classification=classification, regions=regions, image=np.ones((2, 2)),
                                        output_dir=args.output, image_name=os.path.basename(i))


if __name__ == "__main__":
    main()
