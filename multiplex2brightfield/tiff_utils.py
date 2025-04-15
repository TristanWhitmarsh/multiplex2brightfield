import os
import numpy as np
from tqdm import tqdm
import tifffile
from lxml import etree
from utils import maybe_cleanup


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


def get_normalisation_values_from_center_crop(filename, crop_size=8192):
    """
    Compute normalization values for each channel using a center crop from the image.

    This function opens a TIFF file, extracts a centered crop of the specified size from the first page,
    and computes percentiles (20th and 99th by default) for each channel within the crop. The computed 
    percentiles are then used to generate scale and offset values for intensity normalization.

    Args:
        filename (str): Path to the input TIFF file.
        crop_size (int, optional): Size of the square crop taken from the center of the image.
                                   Defaults to 8192.

    Returns:
        dict: A dictionary mapping each channel name to its normalization parameters (with keys 'scale' 
              and 'offset').
    """
    with tifffile.TiffFile(filename) as tif:
        shape = tif.pages[0].shape
        height, width = shape[:2]

        y_start = max(0, (height // 2) - (crop_size // 2))
        y_end = min(height, y_start + crop_size)
        x_start = max(0, (width // 2) - (crop_size // 2))
        x_end = min(width, x_start + crop_size)

        channel_names = []
        if tif.ome_metadata:
            root = etree.fromstring(tif.ome_metadata.encode("utf-8"))
            ns_uri = root.tag.split("}")[0].strip("{")
            channel_elements = root.findall(f".//{{{ns_uri}}}Channel")
            channel_names = [
                ch.get("Name") for ch in channel_elements if ch.get("Name")
            ]

        if not channel_names:
            if len(tif.pages) == 1 and len(shape) >= 3:
                channel_names = [f"Channel_{i}" for i in range(shape[2])]
            else:
                channel_names = [f"Channel_{i}" for i in range(len(tif.pages))]

        norm_values = {}
        for idx, channel_name in enumerate(tqdm(channel_names, desc="Normalising")):
            if len(tif.pages) == 1 and len(shape) >= 3:
                crop_data = tif.pages[0].asarray()[y_start:y_end, x_start:x_end, idx]
            else:
                crop_data = tif.pages[idx].asarray()[y_start:y_end, x_start:x_end]
            p_min = np.percentile(crop_data, 20)
            p_max = np.percentile(crop_data, 99)
            norm_values[channel_name] = {
                "scale": 1.0 if p_max == p_min else 1.0 / (p_max - p_min),
                "offset": 0.0 if p_max == p_min else -p_min / (p_max - p_min),
            }
            del crop_data
            maybe_cleanup()
        return norm_values


def add_pyramids_efficient_memmap(
    input_memmap_path,
    output_path,
    shape,
    dtype=np.uint8,
    num_levels=8,
    tile_size=1024,
    base_metadata=None,
):
    """
    Generate a multi-resolution pyramid from a memory-mapped base image without loading the entire image into memory.

    This function reads a large base image from a memory-mapped file and writes an OME-TIFF with pyramid levels.
    The base image is written first, followed by iteratively downscaled pyramid levels written as sub-files.
    The downscaling is performed in chunks to optimize memory usage.

    Args:
        input_memmap_path (str): Path to the memory-mapped file containing the base image.
        output_path (str): Path where the output OME-TIFF file with pyramids will be saved.
        shape (tuple): A tuple (height, width, 3) describing the dimensions of the base image.
        dtype (numpy.dtype, optional): Data type of the image pixels (default: np.uint8).
        num_levels (int, optional): Number of pyramid levels to generate (default: 8).
        tile_size (int, optional): Tile size for the pyramid tiles (default: 1024).
        base_metadata (dict, optional): Metadata to include with the TIFF file.

    Returns:
        None
    """
    # Open the raw memmap file
    base_data = np.memmap(input_memmap_path, dtype=dtype, mode="r", shape=shape)

    with tifffile.TiffWriter(output_path, bigtiff=True) as tw:
        # Write the full-resolution base image.
        tw.write(
            base_data,
            photometric="rgb",
            metadata=base_metadata,
            tile=(tile_size, tile_size),
            compression="zlib",
            subifds=num_levels - 1,
        )

        current_data = base_data
        for level in range(1, num_levels):
            new_shape = (current_data.shape[0] // 2, current_data.shape[1] // 2, 3)
            # Create a temporary memmap for the downscaled level.
            temp_pyr_path = output_path + f".pyr_level_{level}.dat"
            downscaled = np.memmap(
                temp_pyr_path, dtype=np.float32, mode="w+", shape=new_shape
            )

            max_y = (current_data.shape[0] // 2) * 2
            max_x = (current_data.shape[1] // 2) * 2
            chunk_size = 1024  # adjust as needed
            for y in range(0, max_y, chunk_size):
                end_y = min(y + chunk_size, max_y)
                chunk = (
                    current_data[y:end_y:2, :max_x:2, :].astype(np.float32)
                    + current_data[y:end_y:2, 1:max_x:2, :].astype(np.float32)
                    + current_data[y + 1 : end_y : 2, :max_x:2, :].astype(np.float32)
                    + current_data[y + 1 : end_y : 2, 1:max_x:2, :].astype(np.float32)
                ) / 4.0
                y_scaled = y // 2
                downscaled[
                    y_scaled : y_scaled + chunk.shape[0], : chunk.shape[1], :
                ] = chunk

            downscaled.flush()
            tw.write(downscaled.astype(dtype), tile=(1024, 1024), subfiletype=1)

            # Prepare for next level.
            current_data = np.array(downscaled)
            del downscaled
            os.remove(temp_pyr_path)


def add_pyramids_inmemory(
    base_image, output_path, num_levels=8, tile_size=1024, base_metadata=None
):
    """
    Generate a multi-resolution pyramid from an in-memory base image and save it as an OME-TIFF file.

    This function takes an in-memory base image array and creates pyramid levels by iterative downsampling.
    The resulting pyramid (including the base image) is saved to an OME-TIFF file with the specified 
    tiling and metadata.

    Args:
        base_image (numpy.ndarray): The base image array with shape (height, width, 3).
        output_path (str): Path where the output OME-TIFF file will be saved.
        num_levels (int, optional): Number of pyramid levels to generate (default: 8).
        tile_size (int, optional): Tile size for the pyramid tiles (default: 1024).
        base_metadata (dict, optional): Metadata dictionary to include in the TIFF file.

    Returns:
        None
    """
    with tifffile.TiffWriter(output_path, bigtiff=True) as tw:
        tw.write(
            base_image,
            photometric="rgb",
            metadata=base_metadata,
            tile=(tile_size, tile_size),
            compression="zlib",
            subifds=num_levels - 1,
        )
        current_data = base_image
        for level in range(1, num_levels):
            new_shape = (current_data.shape[0] // 2, current_data.shape[1] // 2, 3)
            downscaled = np.zeros(new_shape, dtype=np.float32)
            max_y = (current_data.shape[0] // 2) * 2
            max_x = (current_data.shape[1] // 2) * 2
            chunk_size = 1024  # still use chunking if desired
            for y in range(0, max_y, chunk_size):
                end_y = min(y + chunk_size, max_y)
                chunk = (
                    current_data[y:end_y:2, :max_x:2, :].astype(np.float32)
                    + current_data[y:end_y:2, 1:max_x:2, :].astype(np.float32)
                    + current_data[y + 1 : end_y : 2, :max_x:2, :].astype(np.float32)
                    + current_data[y + 1 : end_y : 2, 1:max_x:2, :].astype(np.float32)
                ) / 4.0
                y_scaled = y // 2
                downscaled[
                    y_scaled : y_scaled + chunk.shape[0], : chunk.shape[1], :
                ] = chunk
            tw.write(downscaled.astype(base_image.dtype), subfiletype=1)
            current_data = downscaled
