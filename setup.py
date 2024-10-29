from setuptools import setup, find_packages
import pathlib

# Get the long description from the README.md file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='multiplex2brightfield',
    version='0.1.3',
    description='A package to convert a multiplex image to a virtual blightfield image such as H&E or IHC. Both the input and output are in OME-TIFF file format.',
    long_description=long_description,  # Add this to include the README.md content
    long_description_content_type='text/markdown',  # Specify the content type as Markdown
    author='Tristan Whitmarsh',
    author_email='tw401@cam.ac.uk',
    url='https://github.com/TristanWhitmarsh/multiplex2brightfield',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tifffile',
        'scikit-image',
        'numpy2ometiff',
        'csbdeep',
        'scikit-image',
        'keras',
        'pillow',
        'SimpleITK',
        'lxml',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    python_requires='>=3.6',
)
