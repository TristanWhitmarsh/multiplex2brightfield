from setuptools import setup, find_packages
import pathlib

# Get the long description from the README.md file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='multiplex2brightfield',
    version='0.1.13',
    description='A package to convert a multiplex image to a virtual blightfield image such as H&E or IHC. Both the input and output are in OME-TIFF file format.',
    long_description=long_description,  # Add this to include the README.md content
    long_description_content_type='text/markdown',  # Specify the content type as Markdown
    author='Tristan Whitmarsh',
    author_email='tw401@cam.ac.uk',
    url='https://github.com/TristanWhitmarsh/multiplex2brightfield',
    packages=find_packages(),
    install_requires=[
        'tensorflow==2.15.0',
        'numpy==1.26.4',
        'tifffile>=2024.8.30',
        'scikit-image>=0.24.0',
        'numpy2ometiff>=0.1.4',
        'csbdeep>=0.8.1',
        'SimpleITK>=2.4.0',
        'lxml>=5.3.0',
        'requests>=2.32.3',
        'tqdm>=4.66.5',
        'psutil>=6.0.0',
        'protobuf==3.20.3',
    ],
    extras_require={
        'enhance': [
            'keras>=2.10.0',
        ],
        'llm': [
            'openai>=1.52.0',
            'google-generativeai>=0.8.3',
            'anthropic>=0.37.1',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    python_requires='>=3.9',
)
