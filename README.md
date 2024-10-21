
# multiplex2brightfield

`multiplex2brightfield` is a Python package that converts multiplex imaging data (such as imaging mass cytometry) into virtual brightfield images, such as Hematoxylin & Eosin (H&E) or Immunohistochemistry (IHC) images. The input and output are both handled in OME-TIFF file format, allowing seamless integration into bioimaging workflows.

## Features

- **Convert Multiplex Images to Virtual Brightfield**: Convert complex multiplex images to virtual brightfield formats such as H&E.
- **OME-TIFF Input/Output**: The package supports OME-TIFF file format for both input and output, preserving metadata and structure.
- **Marker-based Channel Identification**: Identify relevant channels for nuclear, cytoplasmic, and other staining components (e.g., haematoxylin, eosin, etc.) based on channel names.
- **Customizable Filters**: Apply customizable median and Gaussian filters to different color channels.
- **Histogram Normalization**: Optionally apply histogram normalization to the haematoxylin and eosin channels to adjust contrast.
- **Optional Pyramid Creation**: Generate a multi-resolution pyramid in OME-TIFF format for efficient visualization of large datasets.

## Installation

Clone the repository and install the required dependencies by running the following command:

```bash
pip install multiplex2brightfield
```

## Usage

Here is an example of how to use the `multiplex2brightfield` package to convert a multiplex image to a virtual H&E-stained image:

```python
from multiplex2brightfield import convert

# Input and output file paths
input_filename = "input_image.ome.tiff"
output_filename = "output_virtual_HE.ome.tiff"

# Call the conversion function
convert(
    input_filename=input_filename,
    output_filename=output_filename,
    show_haematoxylin=True,
    show_eosin1=True,
    show_eosin2=True,
    show_blood=True,
    show_marker=False
)
```

### Parameters

- `input_filename` (str): Path to the OME-TIFF file containing the multiplex image.
- `output_filename` (str): Path where the converted virtual brightfield image will be saved in OME-TIFF format.
- `show_haematoxylin`, `show_eosin1`, `show_eosin2`, `show_blood`, `show_marker` (bool): Flags to control which stains appear in the output.
- `histogram_matching` (bool): Enable or disable histogram matching based on a reference image (optional).
- `apply_filter` (bool): Apply filtering to specific channels (optional).

## Customization

The package allows for extensive customization, including adjusting which channels are visualized, applying filters, and performing histogram normalization.

### Example with Histogram Matching

```python
convert(
    input_filename=input_filename,
    output_filename=output_filename,
    reference_filename="reference_image.jpg",
    histogram_matching=True,
)
```

## Contributing

Contributions to `multiplex2brightfield` are welcome! Feel free to fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
