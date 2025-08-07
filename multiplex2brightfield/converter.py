import os  # Needed for os.remove(temp_output_path)
import json  # Used to dump/print JSON (config, etc.)
import math  # Used for math.ceil, math.floor
import time  # Used for timing (time.time())
import concurrent.futures  # Used for ThreadPoolExecutor

import numpy as np  # Used in array operations, np.zeros, np.mean, etc.
import tifffile  # Used to read TIFF files (tif.pages, tif.asarray())
from xml.etree import ElementTree as ET  # Used for ET.fromstring(metadata)
from csbdeep.utils import normalize  # Used for channel normalization
from skimage.filters import gaussian, median, unsharp_mask
from skimage.morphology import disk
from skimage import exposure, io  # Used for histogram equalization, io.imread
import SimpleITK as sitk  # Used for sitk.GetImageFromArray, etc.
from lxml import etree  # Used for etree.fromstring(ome_metadata.encode('utf-8'))

# import configuration_presets
try:
    from . import configuration_presets
except ImportError:
    import configuration_presets
    
from numpy2ometiff import write_ome_tiff

dtype_working = np.float16

# Local modules
try:
    from .utils import (
    maybe_cleanup,
    format_time_remaining,
    resample_rgb_slices,
    find_channels,
    #get_normalization_values,
)
except ImportError:
    from utils import (
    maybe_cleanup,
    format_time_remaining,
    resample_rgb_slices,
    find_channels,
    #get_normalization_values,
)

try:
    from .tiff_utils import (
    get_image_metadata,
    get_normalisation_values_from_center_crop,
    add_pyramids_efficient_memmap,
    add_pyramids_inmemory,
)
except ImportError:
    from tiff_utils import (
    get_image_metadata,
    get_normalisation_values_from_center_crop,
    add_pyramids_efficient_memmap,
    add_pyramids_inmemory,
)

try:
    from .llm_utils import query_llm_for_channels
except ImportError:
    from llm_utils import query_llm_for_channels

def convert_from_file(
    input_filename,
    output_filename=None,
    use_chatgpt=False,
    use_gemini=False,
    use_claude=False,
    input_pixel_size_x=None,
    input_pixel_size_y=None,
    input_physical_size_z=None,
    imagej=False,
    create_pyramid=True,
    compression="zlib",
    Unit="µm",
    downsample_count=4,
    filter_settings=None,
    AI_enhancement=False,
    output_pixel_size_x=None,
    output_pixel_size_y=None,
    output_physical_size_z=None,
    channel_names=None,
    stain="",
    custom_palette=None,
    api_key="",
    config=None,
    normalization_values=None,
    intensity=1.0,
    median_filter_size=0,
    gaussian_filter_sigma=0,
    sharpen_filter_amount=0,
    histogram_normalisation=False,
    clip=None,
    process_tiled=False,
    tile_size=8192,
    use_memmap=False,
    use_subtractive_method=False,
):
    """
    Convert a multiplex image from an OME-TIFF file to a virtual brightfield image.

    This function reads a multiplex image stored in an OME-TIFF file and processes it to generate a 
    virtual brightfield image (e.g., simulating H&E or IHC staining). It supports both full image processing 
    and tiled processing (useful for very large images). Various processing steps are applied including 
    channel identification, normalization, filtering, and optional AI enhancement. Multi-resolution pyramid 
    generation is also supported for efficient visualization.

    Args:
        input_filename (str): Path to the input OME-TIFF multiplex image.
        output_filename (str, optional): Path for saving the output OME-TIFF virtual brightfield image.
        use_chatgpt (bool): Enable ChatGPT-based configuration for channel mapping.
        use_gemini (bool): Enable Gemini-based configuration for channel mapping.
        use_claude (bool): Enable Claude-based configuration for channel mapping.
        input_pixel_size_x (float, optional): X-axis pixel size of the input image. Defaults to 1.
        input_pixel_size_y (float, optional): Y-axis pixel size of the input image. Defaults to 1.
        input_physical_size_z (float, optional): Z-axis physical size of the input image. Defaults to 1.
        imagej (bool): Flag to indicate if ImageJ compatibility is required.
        create_pyramid (bool): If True, generate multi-resolution pyramid levels in the output image.
        compression (str): Compression method to use (default "zlib").
        Unit (str): Unit for the pixel sizes (default "µm").
        downsample_count (int): Number of pyramid levels to generate.
        filter_settings (dict, optional): Custom filter settings overriding defaults.
        AI_enhancement (bool): If True, apply deep learning-based enhancement to the output image.
        output_pixel_size_x (float, optional): Desired output pixel size in X direction.
        output_pixel_size_y (float, optional): Desired output pixel size in Y direction.
        output_physical_size_z (float, optional): Desired output physical size in Z direction.
        channel_names (list of str, optional): List of channel names extracted from the image.
        stain (str): Name of the stain preset to use (e.g., "H&E", "IHC").
        custom_palette (optional): Custom color palette for stain simulation.
        api_key (str): API key for the LLM service used for channel configuration.
        config (dict, optional): Custom configuration dictionary for the stain.
        normalization_values (dict, optional): Precomputed normalization values for the channels.
        intensity (float): Scaling factor for the intensity of stain components.
        median_filter_size (int): Size of the median filter kernel for noise reduction.
        gaussian_filter_sigma (float): Sigma value for Gaussian smoothing.
        sharpen_filter_amount (float): Amount of sharpening to apply.
        histogram_normalisation (bool): If True, perform histogram normalization on channels.
        clip (tuple or None): Value range for clipping channel intensities.
        process_tiled (bool): If True, process the image in smaller tiles.
        tile_size (int): Size (in pixels) for each tile when processing tiled.
        use_memmap (bool): If True, employ memory mapping for handling large images.

    Returns:
        numpy.ndarray: The processed virtual brightfield image data, transposed appropriately for OME-TIFF writing.
    """
    if input_pixel_size_x is None:
        input_pixel_size_x = 1
    if input_pixel_size_y is None:
        input_pixel_size_y = 1
    if input_physical_size_z is None:
        input_physical_size_z = 1
    
    if process_tiled:
        with tifffile.TiffFile(input_filename) as tif:
            shape = tif.series[0].shape
            if len(shape) == 4:
                n_z, n_channels, height, width = shape
            elif len(shape) == 3:
                n_z = 1
                n_channels, height, width = shape
            else:
                raise ValueError("Unsupported image shape for tiling.")

            pixel_sizes, channel_names = get_image_metadata(tif, False, n_channels)

        print(
            f"Image dimensions: {width}x{height} pixels, Z-sections: {n_z}, Channels: {n_channels}, Pixel sizes: {pixel_sizes}"
        )
        
        if output_pixel_size_x is None:
            output_pixel_size_x = pixel_sizes["x"]
        if output_pixel_size_y is None:
            output_pixel_size_y = pixel_sizes["y"]

        scale_x = pixel_sizes["x"] / output_pixel_size_x
        scale_y = pixel_sizes["y"] / output_pixel_size_y
        scaled_width  = math.ceil(width  * scale_x)
        scaled_height = math.ceil(height * scale_y)

        n_tiles_x = math.ceil(scaled_width / tile_size)
        n_tiles_y = math.ceil(scaled_height / tile_size)
        total_tiles = n_z * n_tiles_y * n_tiles_x

        # Get normalization values only once (could adapt for Z-dependent normalization if needed)
        norm_values = get_normalisation_values_from_center_crop(input_filename)

        # Conditional allocation of the output array (use memmap for large data)
        if use_memmap:
            print("Generating memmap")
            temp_output_path = "temp_output_multiz.dat"
            output_array = np.memmap(
                temp_output_path,
                dtype=np.uint8,
                mode="w+",
                shape=(n_z, scaled_height, scaled_width, 3),
            )
            output_array[:] = 0
        else:
            output_array = np.zeros((n_z, scaled_height, scaled_width, 3), dtype=np.uint8)

        
        processed_tiles = 0
        processing_start_time = time.time()
        
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                tile_data = process_single_tile(
                    tx,
                    ty,
                    tile_size,
                    height,
                    width,
                    input_filename,
                    n_z,              # NEW: number of z-slices
                    n_channels,
                    channel_names,
                    config,
                    norm_values,
                    pixel_sizes,
                    output_pixel_size_x,
                    output_pixel_size_y,
                    False,            # multi_page
                )  # should return shape: (Z, 3, tile_h, tile_w)

                # Write all Z slices into output_array
                for z in range(n_z):
                    tile_z = tile_data[z].transpose(1, 2, 0)  # (tile_h, tile_w, 3)

                    y_start = ty * tile_size
                    x_start = tx * tile_size
                    
                    h, w = tile_z.shape[:2]
                    y_end = y_start + h
                    x_end = x_start + w

                    output_array[z, y_start:y_end, x_start:x_end, :] = tile_z

                processed_tiles += 1
                elapsed_time = time.time() - processing_start_time
                tiles_per_second = processed_tiles / elapsed_time if elapsed_time > 0 else 0
                remaining_tiles = total_tiles - processed_tiles
                estimated_time_remaining = (
                    remaining_tiles / tiles_per_second if tiles_per_second > 0 else 0
                )

                print(
                    f"\rProgress: {processed_tiles}/{total_tiles} tiles "
                    f"({processed_tiles / total_tiles * 100:.1f}%) - "
                    f"Speed: {tiles_per_second:.2f} tiles/sec - "
                    f"Est. remaining: {format_time_remaining(estimated_time_remaining)}",
                    end="",
                    flush=True,
                )

                del tile_data
                maybe_cleanup()

        if use_memmap:
            output_array.flush()

        # Prepare metadata for the base image.
        base_metadata = {
            "PhysicalSizeX": output_pixel_size_x,
            "PhysicalSizeY": output_pixel_size_y,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
        }

        print("\nWriting pyramid levels...")
        if use_memmap:
            add_pyramids_efficient_memmap(
                temp_output_path,              # not base_image
                output_path=output_filename,
                shape=output_array.shape,      # <--- add this line
                num_levels=downsample_count,
                tile_size=1024,
                pixel_size_x=output_pixel_size_x,
                pixel_size_y=output_pixel_size_y,
                physical_size_z=output_physical_size_z,
                Unit="µm"
            )
        else:
            base_image = np.array(output_array)
            add_pyramids_inmemory(
                base_image,
                output_path=output_filename,
                num_levels=downsample_count,
                tile_size=1024,
                pixel_size_x=output_pixel_size_x,
                pixel_size_y=output_pixel_size_y,
                physical_size_z=output_physical_size_z,
                Unit="µm"
            )

        if use_memmap:
            del output_array
            os.remove(temp_output_path)
        maybe_cleanup()

    else:
        # Load the TIFF file and get the metadata
        with tifffile.TiffFile(input_filename) as tif:
            imc_image = tif.asarray()
            metadata = tif.pages[0].tags["ImageDescription"].value
            try:
                ome_metadata = tif.ome_metadata
                if ome_metadata:  # Ensure metadata is not None
                    # Parse XML metadata using lxml
                    root = etree.fromstring(ome_metadata.encode("utf-8"))

                    # Find the Pixels element using a wildcard for the namespace
                    pixels = root.find(".//{*}Pixels")

                    if pixels is not None:
                        # Extracting the attributes
                        input_pixel_size_x = float(pixels.get("PhysicalSizeX", 1))
                        input_pixel_size_y = float(pixels.get("PhysicalSizeY", 1))
                        input_physical_size_z = float(pixels.get("PhysicalSizeZ", 1))

                    # Extract channel names
                    ns_uri = root.tag.split("}")[0].strip("{")
                    ns = {"ome": ns_uri}

                    root = ET.fromstring(metadata)
                    channel_elements = root.findall(".//ome:Channel", ns)

                    if channel_names is None:  # Use provided channel_names if available
                        channel_names = [
                            channel.get("Name")
                            for channel in channel_elements
                            if channel.get("Name")
                        ]
                else:
                    print(f"Warning: OME metadata is missing in {input_filename}.")
            except Exception as e:
                print(
                    f"Warning: Failed to extract metadata for {input_filename}. Error: {e}"
                )
                # Use default values for pixel sizes and an empty channel list if no metadata
                if channel_names is None:
                    channel_names = channel_names if channel_names else []

        # Elegant printing of the input and output pixel sizes
        print(f"Input Pixel Size X: {input_pixel_size_x}")
        print(f"Input Pixel Size Y: {input_pixel_size_y}")
        print(f"Input Physical Size Z: {input_physical_size_z}")
        print(f"Output Pixel Size X: {output_pixel_size_x}")
        print(f"Output Pixel Size Y: {output_pixel_size_y}")
        print(f"Output Physical Size Z: {output_physical_size_z}")

        if imc_image.ndim == 3:
            imc_image = np.expand_dims(imc_image, axis=0)
            print(imc_image.shape)  # The shape will now be (1, height, width, channels)

        print("Data size: ", imc_image.shape)
        print("Image size: ", imc_image.shape[2:4])
        print("Number of channels: ", imc_image.shape[1])
        print("Number of slices: ", imc_image.shape[0])
        # print("Channel names: ", channel_names)

        # if normalization_values is None:
        #     normalization_values = get_normalization_values(
        #         imc_image,
        #         channel_names,
        #         percentile_min=all_channel_normalize_percentage_min,
        #         percentile_max=all_channel_normalize_percentage_max,
        #     )

        return convert(
            imc_image,
            output_filename=output_filename,
            input_pixel_size_x=input_pixel_size_x,
            input_pixel_size_y=input_pixel_size_y,
            input_physical_size_z=input_physical_size_z,
            use_chatgpt=use_chatgpt,
            use_gemini=use_gemini,
            use_claude=use_claude,
            imagej=imagej,
            create_pyramid=create_pyramid,
            compression=compression,
            Unit=Unit,
            downsample_count=downsample_count,
            filter_settings=filter_settings,
            AI_enhancement=AI_enhancement,
            output_pixel_size_x=output_pixel_size_x,
            output_pixel_size_y=output_pixel_size_y,
            output_physical_size_z=output_physical_size_z,
            channel_names=channel_names,
            stain=stain,
            custom_palette=custom_palette,
            api_key=api_key,
            config=config,
            normalization_values=normalization_values,
            intensity=intensity,
            median_filter_size=median_filter_size,
            gaussian_filter_sigma=gaussian_filter_sigma,
            sharpen_filter_amount=sharpen_filter_amount,
            histogram_normalisation=histogram_normalisation,
            clip=clip,
            use_subtractive_method = use_subtractive_method,
        )


def convert(
    imc_image,
    output_filename,
    use_chatgpt=False,
    use_gemini=False,
    use_claude=False,
    input_pixel_size_x=1,
    input_pixel_size_y=1,
    input_physical_size_z=1,
    imagej=False,
    create_pyramid=True,
    compression="zlib",
    Unit="µm",
    downsample_count=8,
    filter_settings=None,
    AI_enhancement=False,
    output_pixel_size_x=None,
    output_pixel_size_y=None,
    output_physical_size_z=None,
    channel_names=None,
    stain="",
    custom_palette=None,
    api_key="",
    config=None,
    normalization_values=None,
    intensity=1.0,
    median_filter_size=0,
    gaussian_filter_sigma=0,
    sharpen_filter_amount=0,
    histogram_normalisation=False,
    clip=None,
    use_subtractive_method=False,
):
    """
    Convert a multiplex image (as an array) to a virtual brightfield image.

    This function takes in the multiplex image data (typically as a NumPy array) along with various 
    processing parameters and configuration options, applies channel mapping, filtering, normalization, 
    and optional AI-enhancement to produce a virtual brightfield image. The output image is prepared 
    for saving in OME-TIFF format.

    Args:
        imc_image (numpy.ndarray): The multiplex image as a NumPy array, typically of shape 
            (n_slices, n_channels, height, width).
        output_filename (str): Path where the output virtual brightfield image will be saved.
        use_chatgpt (bool): Enable configuration via ChatGPT-based LLM for channel mapping.
        use_gemini (bool): Enable configuration via Gemini-based LLM for channel mapping.
        use_claude (bool): Enable configuration via Claude-based LLM for channel mapping.
        input_pixel_size_x (float): Pixel size in the X direction of the input image.
        input_pixel_size_y (float): Pixel size in the Y direction of the input image.
        input_physical_size_z (float): Physical size in the Z direction of the input image.
        imagej (bool): Flag for adapting output for ImageJ.
        create_pyramid (bool): If True, generate a multi-resolution pyramid in the output.
        compression (str): Compression method used for the output image.
        Unit (str): Unit of measurement for physical sizes (e.g., "µm").
        downsample_count (int): Number of pyramid levels to generate.
        filter_settings (dict, optional): Custom filtering settings.
        AI_enhancement (bool): If True, apply deep learning-based enhancement to the output.
        output_pixel_size_x (float, optional): Desired output pixel size in the X direction.
        output_pixel_size_y (float, optional): Desired output pixel size in the Y direction.
        output_physical_size_z (float, optional): Desired output physical size in the Z direction.
        channel_names (list of str, optional): List of channel names from the multiplex image.
        stain (str): The stain preset name to use (e.g., "H&E", "IHC").
        custom_palette (optional): Custom color palette for the stain simulation.
        api_key (str): API key for LLM-based configuration.
        config (dict, optional): Custom configuration dictionary for stain simulation.
        normalization_values (dict, optional): Precomputed normalization values.
        intensity (float): Intensity scaling factor for the stain components.
        median_filter_size (int): Size of the median filter kernel.
        gaussian_filter_sigma (float): Sigma for Gaussian smoothing.
        sharpen_filter_amount (float): Amount of sharpening to apply.
        histogram_normalisation (bool): If True, perform histogram normalization.
        clip (tuple or None): Clipping range for intensity values.

    Returns:
        numpy.ndarray: The processed virtual brightfield image data, transposed for OME-TIFF output.
    """

    # if reference_filename is None:
    #     reference_filename = []
    
    if config is None and not (use_chatgpt or use_gemini or use_claude):
        config = configuration_presets.GetConfiguration(stain)

    if use_chatgpt or use_gemini or use_claude:
        if config is None and stain != "":
            config = {
                "name": stain,
            }

        config = query_llm_for_channels(
            config,
            channel_names,
            intensity=intensity,
            median_filter_size=median_filter_size,
            gaussian_filter_sigma=gaussian_filter_sigma,
            sharpen_filter_amount=sharpen_filter_amount,
            histogram_normalisation=histogram_normalisation,
            clip=clip,
            all_channel_normalize_percentage_min=all_channel_normalize_percentage_min,
            all_channel_normalize_percentage_max=all_channel_normalize_percentage_max,
            use_chatgpt=use_chatgpt,
            use_gemini=use_gemini,
            use_claude=use_claude,
            api_key=api_key,
        )
        print(f"{json.dumps(config, indent=4)}\n\n")

    # stain_name, config = next(iter(config.items()))

    # print(f"Stain name: {stain_name}")

    if not output_pixel_size_x:
        output_pixel_size_x = input_pixel_size_x
    if not output_pixel_size_y:
        output_pixel_size_y = input_pixel_size_y
    if not output_physical_size_z:
        output_physical_size_z = input_physical_size_z

    # Channel names provided
    # channel_names_string = ", ".join(channel_names)
    # prompt_text = config.get("prompt", "")
    # prompt = "Consider the following markers in a multiplexed image: " + channel_names_string + prompt_text

    # print("Channel names: ", channel_names_string)

    # Build a background color array.
    background_key = "background" if "background" in config else "Background"
    background_color = (
        np.array(
            [
                config[background_key]["color"]["R"],
                config[background_key]["color"]["G"],
                config[background_key]["color"]["B"],
            ]
        )
        / 255.0
    )

    # Create working arrays.
    white_image = np.full(
        (imc_image.shape[0], imc_image.shape[2], imc_image.shape[3], 3),
        background_color,
        dtype=dtype_working,
    )
    base_image = np.full_like(white_image, background_color.astype(dtype_working))
    transmission = np.full_like(white_image, background_color.astype(dtype_working))

    # Process each configuration entry.
    for key, value in config["components"].items():
        # print(f"Processing component {key}")
        
        if not isinstance(value, dict):
            continue
        if "targets" not in value:
            continue
        if key.lower() == "background":
            continue

        # Get channels matching the config targets.
        channel_list = find_channels(channel_names, value["targets"])
        if not channel_list:
            continue

        # print(channel_list)
        # print("Shape imc_image:", imc_image.shape)

        # Normalize channels (vectorized if normalization_values is provided).
        if normalization_values:
            z_count = imc_image.shape[0]
            images = []

            norm_slices = []

            for i in range(z_count):
                norm_channels = []
                for ch in channel_list:
                    idx = channel_names.index(ch)
                    # norm_channel = normalize(
                    #     imc_image[i, idx, :, :],  # shape (H, W)
                    #     val1,
                    #     val2,
                    # )
                    # print(f"Normalization keys for Z={i}: {list(normalization_values[i].keys())}")
                    # print(f"Channel list used: {channel_list}")
                    # print(f"Normalization keys for Z={i}: {list(normalization_values[i].keys())}")

                    scale = normalization_values[i][ch]["scale"]
                    offset = normalization_values[i][ch]["offset"]
                    
                    # print(f"i {i} ch {ch} scale {scale} offset {offset}")                    

                    norm_channel = np.clip(imc_image[i, idx, :, :] * scale + offset, 0, 1).astype(np.float16)
                    
                    if norm_channel.ndim == 3 and norm_channel.shape[0] == 1:
                        norm_channel = norm_channel[0]

                    norm_channel = np.clip(norm_channel, 0, 1).astype(np.float16)
                    norm_channels.append(norm_channel)

                norm_channels = np.array(norm_channels)  # shape: (C, H, W)
                avg_slice = np.mean(norm_channels, axis=0).astype(np.float16)  # shape: (H, W)
                norm_slices.append(avg_slice)

            image = np.stack(norm_slices, axis=0)  # shape: (Z, H, W)
            
        else:
            z_count = imc_image.shape[0]
            images = []
            normalize_percentage_min = value["normalize_percentage_min"]
            normalize_percentage_max = value["normalize_percentage_max"]
            # print(f"normalize_percentage_min {val1}, normalize_percentage_max {val2}")
            # print(f"channel_list: {channel_list}")
            
            norm_slices = []

            for i in range(z_count):
                norm_channels = []
                for ch in channel_list:
                    idx = channel_names.index(ch)
                    
                    
                    val1 = np.percentile(imc_image[i, idx, :, :], normalize_percentage_min)
                    val2 = np.percentile(imc_image[i, idx, :, :], normalize_percentage_max)

                    scale = 1.0 / (val2 - val1)
                    offset = -val1 / (val2 - val1)
                    
                    # print(f"i {i} ch {ch} scale {scale} offset {offset}")    
                    
                    norm_channel = np.clip(imc_image[i, idx, :, :] * scale + offset, 0, 1).astype(np.float16)
                    # norm_channel = normalize(
                    #     imc_image[i, idx, :, :],  # shape (H, W)
                    #     val1,
                    #     val2,
                    # )
                    
                    if norm_channel.ndim == 3 and norm_channel.shape[0] == 1:
                        norm_channel = norm_channel[0]

                    norm_channel = np.clip(norm_channel, 0, 1).astype(np.float16)
                    norm_channels.append(norm_channel)

                norm_channels = np.array(norm_channels)  # shape: (C, H, W)
                avg_slice = np.mean(norm_channels, axis=0).astype(np.float16)  # shape: (H, W)
                norm_slices.append(avg_slice)

            image = np.stack(norm_slices, axis=0)

        # Convert to float32 before filtering
        image = image.astype(np.float32)
        
        # Apply median filter if needed.
        if value["median_filter_size"] > 0:
            temp = [
                median(image[i, ...], disk(value["median_filter_size"]))
                for i in range(image.shape[0])
            ]
            image = np.stack(temp, axis=0)
            del temp
            maybe_cleanup()

        # Apply gaussian filter if needed.
        if value["gaussian_filter_sigma"] > 0:
            temp = [
                gaussian(image[i, ...], sigma=value["gaussian_filter_sigma"])
                for i in range(image.shape[0])
            ]
            image = np.stack(temp, axis=0)
            del temp
            maybe_cleanup()

        if (
            value.get("sharpen_filter_amount") is not None
            and value["sharpen_filter_amount"] > 0
        ):
            temp = [
                unsharp_mask(
                    image[i, ...],
                    radius=value["sharpen_filter_radius"],
                    amount=value["sharpen_filter_amount"],
                )
                for i in range(image.shape[0])
            ]
            image = np.stack(temp, axis=0)
            del temp
            maybe_cleanup()

        # Apply histogram equalization if enabled.
        if value["histogram_normalisation"]:
            kernel_size = (50, 50)
            clip_limit = 0.02
            nbins = 256
            temp = [
                exposure.equalize_adapthist(
                    image[i, ...],
                    kernel_size=kernel_size,
                    clip_limit=clip_limit,
                    nbins=nbins,
                )
                for i in range(image.shape[0])
            ]
            image = np.stack(temp, axis=0)
            del temp
            maybe_cleanup()

        if value.get("clip") is not None:
            image = np.clip(image, value["clip"][0], value["clip"][1])
            image = normalize(image, 0, 100)

        # Adjust intensity.
        if use_subtractive_method:
            image *= 0.6 * (value["intensity"])
        else:
            image *= value["intensity"]
            
        # print(f"key {key}")
        # print(f"value {value}")
        
        # Calculate exponential attenuation.
        color = (
            np.array([value["color"]["R"], value["color"]["G"], value["color"]["B"]]) / 255.0
        )
        component_color = background_color - color
        
        if use_subtractive_method:
            transmission -= (
                component_color[np.newaxis, np.newaxis, np.newaxis, :] * image[..., np.newaxis]
            )
        else:
            transmission *= np.exp(
                -component_color[np.newaxis, np.newaxis, np.newaxis, :] * image[..., np.newaxis]
            )
        
        # Delete image after using it.
        del image
        maybe_cleanup()

    
    # Apply the computed transmission to base_image.
    def apply_exponential(i):
        base_image[..., i] = transmission[..., i]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(apply_exponential, range(3)))
    maybe_cleanup()

    # Clip and convert to uint8.
    base_image = np.clip(base_image, 0, 1)
    base_image_uint8 = (base_image * 255).astype(np.uint8)
    # Free unneeded arrays.
    del base_image, transmission
    maybe_cleanup()

    # Resample if needed.
    if (
        output_pixel_size_x != input_pixel_size_x
        or output_pixel_size_y != input_pixel_size_y
    ):
        # print("Resampling")

        if AI_enhancement:
            interpolation = sitk.sitkNearestNeighbor
        else:
            interpolation = sitk.sitkLinear

        base_image_uint8 = resample_rgb_slices(
            base_image_uint8,
            input_pixel_size_x,
            input_pixel_size_y,
            output_pixel_size_x,
            output_pixel_size_y,
            interpolation=interpolation,
        )
        maybe_cleanup()

    # Apply AI enhancement if requested.
    if AI_enhancement:
        print("Enhancing image")
        try:
            from .enhancement import EnhanceBrightfield
        except ImportError:
            from enhancement import EnhanceBrightfield
        
        base_image_uint8 = np.squeeze(base_image_uint8, axis=0)
        base_image_uint8 = EnhanceBrightfield(base_image_uint8)
        base_image_uint8 = np.expand_dims(base_image_uint8, axis=0)
        maybe_cleanup()

    # Apply histogram matching if a reference is provided.
    # if reference_filename:
    #     print("Applying histogram matching")
    #     reference_image = io.imread(reference_filename)
    #     matched_images = []
    #     for i in range(base_image_uint8.shape[0]):
    #         matched_image = exposure.match_histograms(
    #             base_image_uint8[i], reference_image, channel_axis=-1
    #         )
    #         matched_images.append(matched_image)
    #     base_image_uint8 = np.array(matched_images)
    #     del matched_images, reference_image
    #     maybe_cleanup()

    # Final transposition.
    base_image_uint8_transpose = np.transpose(base_image_uint8, (0, 3, 1, 2))
    del base_image_uint8
    maybe_cleanup()

    # Write the output file if requested.
    if output_filename:
        if output_filename.lower().endswith(("ome.tif", "ome.tiff")):
            write_ome_tiff(
                data=base_image_uint8_transpose,
                output_filename=output_filename,
                pixel_size_x=output_pixel_size_x,
                pixel_size_y=output_pixel_size_y,
                physical_size_z=output_physical_size_z,
                Unit="µm",
                imagej=False,
                create_pyramid=True,
                compression="zlib",
                downsample_count=downsample_count,
            )
            print("The OME-TIFF file has been successfully written.")
        else:
            # For non-OME, perform an extra transposition.
            base_image_uint8_transpose = np.transpose(
                base_image_uint8_transpose, (0, 3, 2, 1)
            ).astype(np.uint8)
            sitk_image = sitk.GetImageFromArray(
                base_image_uint8_transpose, isVector=True
            )
            image_size = sitk_image.GetSize()
            print(f"Image size: {image_size}")
            sitk_image.SetSpacing(
                [output_pixel_size_x, output_pixel_size_y, output_physical_size_z]
            )
            sitk_image.SetOrigin([0.0, 0.0, 0.0])
            sitk.WriteImage(sitk_image, output_filename)
            print("The file has been successfully written.")
            del sitk_image
            maybe_cleanup()

    # Delete remaining large arrays.
    del imc_image, white_image
    maybe_cleanup()

    return base_image_uint8_transpose


def process_single_tile(
    tx,
    ty,
    tile_size,
    height,
    width,
    filename,
    n_z,
    n_channels,
    channel_names,
    config,
    norm_values,
    pixel_sizes,
    output_pixel_size_x,
    output_pixel_size_y,
    multi_page,
):
    import tifffile
    import numpy as np
    import math
    import gc

    scale_x = pixel_sizes["x"] / output_pixel_size_x
    scale_y = pixel_sizes["y"] / output_pixel_size_y

    y_start_out = ty * tile_size
    x_start_out = tx * tile_size

    # Use ceil to match the same logic as convert_from_file
    scaled_height = math.ceil(height * scale_y)
    scaled_width  = math.ceil(width  * scale_x)

    y_end_out = min((ty + 1) * tile_size, scaled_height)
    x_end_out = min((tx + 1) * tile_size, scaled_width)
    target_height = y_end_out - y_start_out
    target_width  = x_end_out - x_start_out

    input_y_start = int(math.floor(y_start_out / scale_y))
    input_x_start = int(math.floor(x_start_out / scale_x))
    input_y_end   = int(math.ceil(min(y_end_out, scaled_height) / scale_y))
    input_x_end   = int(math.ceil(min(x_end_out,  scaled_width)  / scale_x))

    if input_y_end <= input_y_start or input_x_end <= input_x_start:
        return np.zeros((n_z, 3, target_height, target_width), dtype=np.uint8)

    try:
        used_channel_indices = list(range(n_channels))
        tile_channels = []

        with tifffile.TiffFile(filename) as tif:
            shape = tif.series[0].shape

            for z in range(n_z):
                z_channels = []
                for c in used_channel_indices:
                    if len(shape) == 4:  # (Z, C, Y, X)
                        page_index = z * n_channels + c
                    elif len(shape) == 3:  # (C, Y, X)
                        page_index = c
                    else:
                        raise ValueError(f"Unsupported TIFF shape: {shape}")

                    try:
                        crop = tif.pages[page_index].asarray(
                            key=(slice(input_y_start, input_y_end), slice(input_x_start, input_x_end))
                        )
                    except Exception:
                        full_channel = tif.pages[page_index].asarray()
                        crop = full_channel[input_y_start:input_y_end, input_x_start:input_x_end]
                        del full_channel

                    z_channels.append(crop)
                    del crop
                    gc.collect()

                tile_channels.append(np.stack(z_channels, axis=0))  # (C, H, W)

        tile = np.stack(tile_channels, axis=0)  # (Z, C, H, W)

        processed = convert(
            tile,
            output_filename=None,
            input_pixel_size_x=pixel_sizes["x"],
            input_pixel_size_y=pixel_sizes["y"],
            output_pixel_size_x=output_pixel_size_x,
            output_pixel_size_y=output_pixel_size_y,
            channel_names=channel_names,
            config=config,
            create_pyramid=False,
            normalization_values=norm_values,
            AI_enhancement=False,
        )

        # Allocate output at nominal size and safely copy data
        result = np.zeros(
            (n_z, 3, target_height, target_width),
            dtype=processed.dtype,
        )

        for z in range(n_z):
            tile_rgb = processed[z]  # (3, H_real, W_real)

            h = min(tile_rgb.shape[1], target_height)
            w = min(tile_rgb.shape[2], target_width)

            result[z, :, :h, :w] = tile_rgb[:, :h, :w]

        del processed, tile, tile_channels
        gc.collect()

        return result  # (Z, 3, H, W)

    except Exception as e:
        print(f"Error processing tile at (tx={tx}, ty={ty}): {e}")
        raise

