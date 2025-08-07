import os
import numpy as np
from tqdm import tqdm
import tifffile
import skimage.measure
from lxml import etree
try:
    from .utils import maybe_cleanup
except ImportError:
    from utils import maybe_cleanup
import gc



def get_image_metadata(tif, multi_page, n_channels):
    """
    Retrieve pixel size information and channel names from an OME-TIFF file's metadata.

    This function extracts the pixel sizes (x, y, z) and channel names from the OME metadata
    of a TIFF file. If the OME metadata is absent or incomplete, default values are used:
    pixel size defaults to 1.0 and channel names are generated as "Channel_0", "Channel_1", etc.

    Args:
        tif (tifffile.TiffFile): An open TIFF file object.
        multi_page (bool): Indicates whether the TIFF file contains multiple pages (one per channel)
                           or a single page with multiple channels.
        n_channels (int): The number of channels in the image.

    Returns:
        tuple:
            - pixel_sizes (dict): A dictionary with keys "x", "y", and "z" mapping to float values.
            - channel_names (list of str): A list of channel names.
    """
    pixel_sizes = {"x": 1.0, "y": 1.0, "z": 1.0}
    channel_names = []

    if tif.ome_metadata:
        root = etree.fromstring(tif.ome_metadata.encode("utf-8"))
        pixels = root.find(".//{*}Pixels")
        if pixels is not None:
            pixel_sizes = {
                "x": float(pixels.get("PhysicalSizeX", 1.0)),
                "y": float(pixels.get("PhysicalSizeY", 1.0)),
                "z": float(pixels.get("PhysicalSizeZ", 1.0)),
            }
        ns_uri = root.tag.split("}")[0].strip("{")
        channel_elements = root.findall(f".//{{{ns_uri}}}Channel")
        channel_names = [ch.get("Name") for ch in channel_elements if ch.get("Name")]

    if not channel_names:
        channel_names = [f"Channel_{i}" for i in range(n_channels)]

    # print(f"Channel names: {channel_names}")
    return pixel_sizes, channel_names


def get_normalisation_values_from_center_crop(filename, crop_size=16384):
    import tifffile
    from lxml import etree
    from tqdm import tqdm
    import numpy as np
    import gc

    with tifffile.TiffFile(filename) as tif:
        shape = tif.series[0].shape
        ndim = len(shape)

        if ndim == 4:
            n_z, n_channels, height, width = shape
        elif ndim == 3:
            n_z = 1
            n_channels, height, width = shape
        else:
            raise ValueError(f"Unsupported TIFF shape: {shape}")

        y_start = max(0, (height // 2) - (crop_size // 2))
        y_end   = min(height, y_start + crop_size)
        x_start = max(0, (width  // 2) - (crop_size // 2))
        x_end   = min(width,  x_start + crop_size)

        # Extract channel names
        channel_names = []
        if tif.ome_metadata:
            root = etree.fromstring(tif.ome_metadata.encode("utf-8"))
            ns_uri = root.tag.split("}")[0].strip("{")
            channel_elements = root.findall(f".//{{{ns_uri}}}Channel")
            channel_names = [ch.get("Name") for ch in channel_elements if ch.get("Name")]

        if not channel_names:
            channel_names = [f"Channel_{i}" for i in range(n_channels)]

        norm_values = {}

        for z_index in range(n_z):
            norm_values[z_index] = {}
            for c, channel_name in enumerate(tqdm(channel_names, desc=f"Z={z_index}")):
                try:
                    if ndim == 4:
                        page_index = z_index * n_channels + c
                    elif ndim == 3:
                        page_index = c
                    else:
                        raise RuntimeError("Unexpected image dimensionality.")

                    page = tif.pages[page_index]
                    full_channel = page.asarray()
                    crop = full_channel[y_start:y_end, x_start:x_end]
                    del full_channel

                except Exception as e:
                    raise RuntimeError(f"Failed to extract crop for channel {c}, slice {z_index}: {e}")

                p_min = np.percentile(crop, 10)
                p_max = np.percentile(crop, 99)
                scale = 1.0 if p_max == p_min else 1.0 / (p_max - p_min)
                offset = 0.0 if p_max == p_min else -p_min / (p_max - p_min)

                norm_values[z_index][channel_name] = {
                    "scale": scale,
                    "offset": offset
                }

                # print(f"Z {z_index} | Channel {channel_name} | scale: {scale:.6f} | offset: {offset:.6f}")
                del crop
                gc.collect()

        return norm_values






def add_pyramids_efficient_memmap(
    input_memmap_path,
    output_path,
    shape,
    num_levels=4,
    tile_size=1024,
    pixel_size_x=1,
    pixel_size_y=1,
    physical_size_z=1,
    Unit="µm"
):
    """
    Write Z-stack RGB OME-TIFF with pyramids (ZYXC axes), from a memory-mapped array.
    """
    import numpy as np
    import skimage.measure
    import tifffile

    data = np.memmap(input_memmap_path, dtype=np.uint8, mode="r", shape=shape)
    z = data.shape[0]

    metadata = {
        'axes': 'ZYXC',
        'PhysicalSizeX': pixel_size_x,
        'PhysicalSizeXUnit': Unit,
        'PhysicalSizeY': pixel_size_y,
        'PhysicalSizeYUnit': Unit,
        'PhysicalSizeZ': physical_size_z,
        'PhysicalSizeZUnit': Unit,
        'Photometric': 'RGB',
        'Planarconfig': 'contig',
    }

    # Build pyramid levels
    pyramid = [np.array(data)]
    for i in range(1, num_levels):
        prev = pyramid[-1]
        out_y = int(np.ceil(prev.shape[1] / 2))
        out_x = int(np.ceil(prev.shape[2] / 2))
        downsampled = np.empty((z, out_y, out_x, 3), dtype=np.uint8)
        for zi in range(z):
            reduced = skimage.measure.block_reduce(
                prev[zi], block_size=(2, 2, 1), func=np.mean
            ).astype(np.uint8)
            downsampled[zi, :reduced.shape[0], :reduced.shape[1], :] = reduced
        pyramid.append(downsampled)

    # Write base image + pyramid levels
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        tif.write(
            pyramid[0],
            photometric="rgb",
            metadata=metadata,
            tile=(tile_size, tile_size),
            compression="zlib",
            subifds=num_levels - 1,
        )
        for i in range(1, num_levels):
            tif.write(
                pyramid[i],
                photometric="rgb",
                subfiletype=1,
                metadata=None,
                tile=(tile_size, tile_size),
                compression="zlib",
            )





def add_pyramids_inmemory(
    base_image,
    output_path,
    num_levels=4,
    tile_size=1024,
    pixel_size_x=1,
    pixel_size_y=1,
    physical_size_z=1,
    Unit="µm"
):
    """
    Write Z-stack RGB OME-TIFF with pyramids (ZYXC axes).
    """
    if base_image.ndim == 3:
        base_image = base_image[np.newaxis, ...]
    assert base_image.ndim == 4 and base_image.shape[-1] == 3, "Input must be (Z, Y, X, 3)"
    z = base_image.shape[0]

    metadata = {
        'axes': 'ZYXC',
        'PhysicalSizeX': pixel_size_x,
        'PhysicalSizeXUnit': Unit,
        'PhysicalSizeY': pixel_size_y,
        'PhysicalSizeYUnit': Unit,
        'PhysicalSizeZ': physical_size_z,
        'PhysicalSizeZUnit': Unit,
        'Photometric': 'RGB',
        'Planarconfig': 'contig',
    }

    # Build all pyramid levels up front
    pyramid = [base_image]
    for i in range(1, num_levels):
        prev = pyramid[-1]
        out_y = int(np.ceil(prev.shape[1] / 2))
        out_x = int(np.ceil(prev.shape[2] / 2))
        downsampled = np.empty((z, out_y, out_x, 3), dtype=np.uint8)
        for zi in range(z):
            reduced = skimage.measure.block_reduce(
                prev[zi], block_size=(2, 2, 1), func=np.mean
            ).astype(np.uint8)
            downsampled[zi, :reduced.shape[0], :reduced.shape[1], :] = reduced
        pyramid.append(downsampled)

    # Write base image with full metadata, pyramid levels with metadata=None
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        tif.write(
            pyramid[0],
            photometric="rgb",
            metadata=metadata,
            tile=(tile_size, tile_size),
            compression="zlib",
            subifds=num_levels-1,
        )
        for i in range(1, num_levels):
            tif.write(
                pyramid[i],
                photometric="rgb",
                subfiletype=1,
                metadata=None,
                tile=(tile_size, tile_size),
                compression="zlib",
            )