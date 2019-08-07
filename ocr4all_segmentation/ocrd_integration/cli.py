import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from .ocrd_segmentation import OCR4AllSegmentation

@click.command()
@ocrd_cli_options
def ocrd_ocr4all_segmentation(*args, **kwargs):
    return ocrd_cli_wrap_processor(OCR4AllSegmentation, *args, **kwargs)