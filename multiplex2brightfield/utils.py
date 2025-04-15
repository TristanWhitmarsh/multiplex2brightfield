import gc
import psutil
import numpy as np  # <-- Add this line
import SimpleITK as sitk
from datetime import timedelta


def maybe_cleanup(threshold_percent=90):
    """
    Check system memory usage and run garbage collection if usage exceeds a specified threshold.

    This function uses psutil to obtain the current system memory usage. If the memory usage
    is greater than or equal to the threshold (default 90%), it triggers garbage collection
    to free unused memory.

    Args:
        threshold_percent (int, optional): The memory usage percentage threshold to trigger cleanup.
                                             Defaults to 90.

    Returns:
        None
    """
    memory_info = psutil.virtual_memory()  # memory usage info
    if memory_info.percent >= threshold_percent:
        print(f"Memory usage is {memory_info.percent}% — calling gc.collect()")
        gc.collect()


def format_time_remaining(seconds):
    """
    Format a number of seconds into a human-readable time string.

    This function converts a duration in seconds into a string formatted as hours, minutes, and seconds.

    Args:
        seconds (int or float): The number of seconds to format.

    Returns:
        str: A string representation of the duration (e.g., "0:01:30").
    """
    return str(timedelta(seconds=int(seconds)))


def resample_rgb_slices(
    image_array,
    input_pixel_size_x,
    input_pixel_size_y,
    output_pixel_size_x,
    output_pixel_size_y,
    interpolation=sitk.sitkLinear,
):
    """
    Resample a stack of RGB image slices to a new pixel resolution.

    This function processes a 4D NumPy array of RGB images (shape: [num_slices, height, width, 3])
    and resamples each slice using SimpleITK's resampling filter based on the ratio between the input and
    desired output pixel sizes. An interpolation method can be specified.

    Args:
        image_array (numpy.ndarray): Input image array with shape (num_slices, height, width, 3).
        input_pixel_size_x (float): Original pixel size in the X-direction.
        input_pixel_size_y (float): Original pixel size in the Y-direction.
        output_pixel_size_x (float): Desired pixel size in the X-direction.
        output_pixel_size_y (float): Desired pixel size in the Y-direction.
        interpolation (SimpleITK.InterpolatorEnum, optional): Interpolation method for resampling.
                                                            Defaults to sitk.sitkLinear.

    Returns:
        numpy.ndarray: Resampled image array.
    """
    num_slices, height, width, channels = image_array.shape
    assert channels == 3, "This function is designed for RGB images with 3 channels."
    resampled_slices = []

    for z in range(num_slices):
        rgb_slice = image_array[z]
        sitk_image = sitk.GetImageFromArray(rgb_slice, isVector=True)
        sitk_image.SetSpacing((input_pixel_size_x, input_pixel_size_y))

        size = sitk_image.GetSize()
        new_size = [
            int(size[0] * (input_pixel_size_x / output_pixel_size_x)),
            int(size[1] * (input_pixel_size_y / output_pixel_size_y)),
        ]

        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetOutputSpacing((output_pixel_size_x, output_pixel_size_y))
        resample_filter.SetSize(new_size)
        resample_filter.SetInterpolator(interpolation)
        resample_filter.SetDefaultPixelValue(0)

        resampled_sitk_image = resample_filter.Execute(sitk_image)
        resampled_slice = sitk.GetArrayFromImage(resampled_sitk_image)
        resampled_slices.append(resampled_slice)

        # Free temporary SimpleITK objects immediately.
        del sitk_image, resampled_sitk_image, resample_filter
        maybe_cleanup()

    resampled_array = np.stack(resampled_slices, axis=0)
    # Optionally, delete resampled_slices if not needed after stacking.
    del resampled_slices
    maybe_cleanup()

    return resampled_array


def find_channels(channel_names, marker_channels):
    """
    Identify channels whose names match specified marker patterns.

    This function cleans both the provided channel names and marker names (removing hyphens,
    spaces, and converting to lowercase) before checking for a partial match. It returns all 
    channel names that contain any of the cleaned marker strings.

    Args:
        channel_names (list of str): List of channel names (e.g., extracted from metadata).
        marker_channels (list of str): List of marker identifiers to match against channel names.

    Returns:
        list of str: Channel names that match any of the provided marker patterns.
    """
    def clean_string(s):
        # Remove hyphens and spaces, convert to lowercase
        return s.replace("-", "").replace(" ", "").lower()

    # Clean all marker patterns
    clean_markers = [clean_string(marker) for marker in marker_channels]

    # Find matches while preserving original channel names
    matches = []
    for channel in channel_names:
        clean_channel = clean_string(channel)
        # Check if any cleaned marker is contained within the cleaned channel name
        if any(marker in clean_channel for marker in clean_markers):
            matches.append(channel)

    return matches


def get_normalization_values(
    image, channel_names, percentile_min=10, percentile_max=90
):
    """
    Calculate normalization factors (scale and offset) for each image channel.

    For each channel in the provided list, this function computes the given percentiles from the 
    image data and calculates a scale and offset to normalize pixel intensities linearly.

    Args:
        image (numpy.ndarray): Multiplex image array with shape (n_slices, n_channels, height, width).
        channel_names (list of str): List of channel names corresponding to the channels in the image.
        percentile_min (int, optional): Lower percentile for normalization. Defaults to 10.
        percentile_max (int, optional): Upper percentile for normalization. Defaults to 90.

    Returns:
        dict: A dictionary mapping each channel name to a dictionary with 'scale' and 'offset' values.
    """
    norm_values = {}
    print(f"Normalizing to {percentile_min} - {percentile_max}")
    for idx, channel in enumerate(channel_names):
        chan_data = image[:, idx, :, :]
        p_min = np.percentile(chan_data, percentile_min)
        p_max = np.percentile(chan_data, percentile_max)
        scale = 1.0 / (p_max - p_min)
        offset = -p_min * scale
        norm_values[channel] = {"scale": scale, "offset": offset}
    return norm_values
