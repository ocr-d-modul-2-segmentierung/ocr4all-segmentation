from typing import NamedTuple


class XmlSettings(NamedTuple):
    author: str = 'TestUser'

    file_basename_suffix = 'xml'
    file_basename_prefix = ''
    pretty_print_xml: bool = True

    encoding = 'utf-8'
