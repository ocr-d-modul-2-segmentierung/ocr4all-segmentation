from setuptools import setup, find_packages

setup(
    name='ocr4all-segmentation',
    version='0.0.1',
    packages=find_packages(),
    license='LGPL-v3.0',
    long_description=open("README.md").read(),
    include_package_data=True,
    author="Alexander Hartelt",
    author_email="alexander.hartelt@informatik.uni-wuerzburg.de",
    url="https://gitlab2.informatik.uni-wuerzburg.de/alh75dg/ocr4all-segmentation.git",
    download_url='https://gitlab2.informatik.uni-wuerzburg.de/alh75dg/ocr4all-segmentation.git',
    entry_points={
        'console_scripts': [
            'ocr4all_segmentation_predict=ocr4all_segmentation.scripts.predict:main',
            'ocrd-ocr4all-segmentation=ocr4all_segmentation.ocrd_integration.cli:ocrd_ocr4all_segmentation',
        ],
    },
    install_requires=open("requirements.txt").read().split(),
    extras_require={
        'tf_cpu': ['tensorflow>=1.6.0'],
        'tf_gpu': ['tensorflow-gpu>=1.6.0'],
    },
    keywords=['OMR', 'Page content detection', 'pixel classifier'],
    data_files=[('', ["requirements.txt"])],
)
