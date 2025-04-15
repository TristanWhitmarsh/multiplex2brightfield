from keras.models import load_model
import os
import numpy as np
import requests
from tqdm import tqdm
from utils import maybe_cleanup


def download_model():
    """
    Download the AI model file if it is not already present.

    This function checks for the existence of the model file ("model.h5") in the current directory.
    If the file is missing, it downloads the model from a specified URL using HTTP streaming and displays
    a progress bar via tqdm. The function returns the file path to the downloaded model.

    Returns:
        str: The file path to "model.h5" (the downloaded model file).
    """
    url = "https://github.com/TristanWhitmarsh/multiplex2brightfield/releases/download/v0.1.3/model.h5"
    model_path = "model.h5"

    if not os.path.exists(model_path):
        print("Downloading model...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with open(model_path, "wb") as f, tqdm(
            desc=model_path,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                bar.update(len(data))
                f.write(data)
    else:
        print("Model already downloaded.")
    return model_path


# At the top of your module, define a module-level variable.
_ai_model = None


def get_ai_model():
    """
    Load and return the AI enhancement model with caching.

    This function checks if the AI model has already been loaded (cached).
    If not, it calls download_model to ensure the model file is available and then loads the model using Keras.
    The loaded model is stored in a global variable, so subsequent calls will return the cached model.

    Returns:
        keras.Model: The loaded AI enhancement model.
    """
    global _ai_model
    if _ai_model is None:
        # Download the model if needed.
        download_model()  # Ensures the file is available.

        _ai_model = load_model("model.h5")
        print("AI model loaded.")
    return _ai_model


def process_tile(tile, model):
    """
    Process a single image tile using the AI model to generate a virtual brightfield output.

    The tile is preprocessed by scaling pixel values from [0, 255] to [-1, 1], then passed through
    the model to generate a processed tile. The output is then post-processed to clip values to [0, 1]
    and converted back to an unsigned 8-bit integer format suitable for RGB display.

    Args:
        tile (numpy.ndarray): A 2D or 3D NumPy array representing an image patch.
        model (keras.Model): The AI model used to process the tile.

    Returns:
        numpy.ndarray: The processed tile as a uint8 RGB image.
    """
    # Preprocess the tile
    tile = (tile - 127.5) / 127.5
    tile = np.expand_dims(tile, 0)

    # Generate the image using the model
    gen_tile = model.predict(tile, verbose=0)

    # Post-process the generated tile
    gen_tile = gen_tile[0]
    gen_tile = (gen_tile + 1) / 2.0
    gen_tile = np.clip(gen_tile, 0, 1)

    # Convert to uint8 for RGB image
    gen_tile_uint8 = (gen_tile * 255).astype(np.uint8)

    return gen_tile_uint8


def process_image_with_tiling(image, model, tile_size=256, step_size=128):
    """
    Process an entire image using a tiling approach with the provided AI model.

    This function divides the input image into overlapping tiles of a specified size and step.
    Each tile is processed using the process_tile function. The center region of each processed tile
    is extracted and then placed into the final output image. This method is used to handle large images
    that might not be processed in one go due to memory constraints.

    Args:
        image (numpy.ndarray): The input RGB image as a NumPy array of shape (height, width, 3).
        model (keras.Model): The AI model used for processing each tile.
        tile_size (int, optional): The size of the square tile to extract (default: 256 pixels).
        step_size (int, optional): The step size between tiles (default: 128 pixels).

    Returns:
        numpy.ndarray: The processed image as a uint8 RGB image.
    """
    h, w, _ = image.shape
    processed_image = np.zeros((h, w, 3))

    for y in range(0, h - tile_size + 1, step_size):
        for x in range(0, w - tile_size + 1, step_size):
            # Extract tile
            tile = image[y : y + tile_size, x : x + tile_size]

            # Process tile
            processed_tile = process_tile(tile, model)

            # Extract center part
            center_y, center_x = tile_size // 4, tile_size // 4
            processed_center = processed_tile[
                center_y : center_y + step_size, center_x : center_x + step_size
            ]

            # Place processed center into the result image
            processed_image[
                y + center_y : y + center_y + step_size,
                x + center_x : x + center_x + step_size,
            ] = processed_center

            # Delete temporary variables to free memory.
            del tile, processed_tile, processed_center
            # Optionally, trigger garbage collection every few iterations.
            # gc.collect()  # (Be cautious: too frequent calls may slow down performance.)

    return processed_image.astype(np.uint8)


def EnhanceBrightfield(input_image):
    """
    Enhance an input image to produce a virtual brightfield image using AI-based processing.

    This function applies AI enhancement to an input RGB image to simulate virtual brightfield staining.
    It first pads the image to reduce border artifacts, processes the padded image using a tiling approach,
    and then removes the padding from the resulting image before returning the final output.

    Args:
        input_image (numpy.ndarray): The input RGB image as a NumPy array.

    Returns:
        numpy.ndarray: The enhanced virtual brightfield image as a uint8 RGB image.
    """
    # Get the cached model.
    model = get_ai_model()

    pad_size = 256
    padded_image = np.pad(
        input_image,
        ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
        mode="reflect",
    )

    # Use the cached model for processing.
    processed_padded_image = process_image_with_tiling(padded_image, model)
    final_image = processed_padded_image[pad_size:-pad_size, pad_size:-pad_size]

    # Free temporary memory.
    del padded_image, processed_padded_image
    maybe_cleanup()

    return final_image
