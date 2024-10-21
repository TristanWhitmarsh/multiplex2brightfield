import numpy as np
import tifffile
import skimage.measure
import os
import re
import numpy as np
import tifffile
from xml.etree       import ElementTree as ET
from csbdeep.utils   import normalize
from matplotlib      import pyplot as plt
from skimage.filters import gaussian, median
from skimage.morphology import disk
import json
from skimage import exposure
from skimage.io import imsave
import SimpleITK as sitk
from numpy2ometiff import write_ome_tiff
from skimage import io


def clean_string(input_string):
    # Remove all non-alphanumeric characters
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    # Convert to lowercase
    cleaned_string = cleaned_string.lower()
    return cleaned_string

# Function to find the channel names containing the marker_channel substrings
def find_channels(channel_names, marker_channels):
    found_channels = []
    cleaned_marker_channels = [clean_string(marker) for marker in marker_channels]
    for name in channel_names:
        cleaned_name = clean_string(name)
        if any(marker in cleaned_name for marker in cleaned_marker_channels):
            found_channels.append(name)
    return found_channels

def create_folder(folder_name):
    # Get the current working directory
    current_directory = os.getcwd()
    # Path for the new directory
    path = os.path.join(current_directory, folder_name)

    # Check if the directory already exists
    if not os.path.exists(path):
        # If it doesn't exist, create it
        os.makedirs(path)
        print(f"Folder '{folder_name}' was created successfully at '{path}'.")
    else:
        # If the directory exists, inform the user
        print(f"Folder '{folder_name}' already exists at '{path}'.")
        

def convert(input_filename, output_filename, reference_filename=[],
    show_haematoxylin = True,
    show_eosin1 = True,
    show_eosin2 = True,
    show_blood = True,
    show_marker = False,
    use_chatgpt = False,
    use_haematoxylin_histogram_normalisation = True,
    use_eosin_histogram_normalisation = True,
    histogram_matching = False,
    channel_names=[], pixel_size_x=1, pixel_size_y=1, physical_size_z=1, imagej=False, create_pyramid=True, compression='zlib', Unit='µm', downsample_count=4, 
    apply_filter = False,
    filter_settings=None,
    api_key = ''):


    # Default filter settings if none are provided
    if apply_filter:
        default_filter_settings = {
            "blue": {"median_filter_size": 1, "gaussian_filter_sigma": 0},
            "pink": {"median_filter_size": 2, "gaussian_filter_sigma": 0.5},
            "purple": {"median_filter_size": 2, "gaussian_filter_sigma": 1},
            "red": {"median_filter_size": 1, "gaussian_filter_sigma": 0},
            "brown": {"median_filter_size": 2, "gaussian_filter_sigma": 1}
        }
    else:
        default_filter_settings = {
            "blue": {"median_filter_size": 0, "gaussian_filter_sigma": 0},
            "pink": {"median_filter_size": 0, "gaussian_filter_sigma": 0},
            "purple": {"median_filter_size": 0, "gaussian_filter_sigma": 0},
            "red": {"median_filter_size": 0, "gaussian_filter_sigma": 0},
            "brown": {"median_filter_size": 0, "gaussian_filter_sigma": 0}
        }

    # Merge user-provided filter settings with the default ones
    if filter_settings is not None:
        for key in default_filter_settings:
            if key in filter_settings:
                default_filter_settings[key].update(filter_settings[key])

    filter_settings = default_filter_settings

    # Extract individual filter settings
    blue_median_filter_size = filter_settings["blue"]["median_filter_size"]
    blue_gaussian_filter_sigma = filter_settings["blue"]["gaussian_filter_sigma"]

    pink_median_filter_size = filter_settings["pink"]["median_filter_size"]
    pink_gaussian_filter_sigma = filter_settings["pink"]["gaussian_filter_sigma"]

    purple_median_filter_size = filter_settings["purple"]["median_filter_size"]
    purple_gaussian_filter_sigma = filter_settings["purple"]["gaussian_filter_sigma"]

    red_median_filter_size = filter_settings["red"]["median_filter_size"]
    red_gaussian_filter_sigma = filter_settings["red"]["gaussian_filter_sigma"]

    brown_median_filter_size = filter_settings["brown"]["median_filter_size"]
    brown_gaussian_filter_sigma = filter_settings["brown"]["gaussian_filter_sigma"]
    
        
    # Default markers
    haematoxylin_list = ['DNA', 'DAPI', 'hoechst'] # 'H3K27me3', 'Ki67' , 'SOX9', 'NFIB'
    pink_eosin_list = ['Col1A1', 'FN', 'VIM', 'aSMA', 'CD31', 'PECAM1', 'Col3A1', 'Col4A1', 'Desmin', 'Fibronectin', 'Laminin', 'Actin', 'eosin', 'stroma']
    purple_eosin_list = ['panCK', 'CAV1', 'AQP1', 'EPCAM', 'CK7', 'CK20', 'E-cadherin', 'P-cadherin', 'MUC1', 'S100', 'SMA', 'epithelium']
    blood_list = ['Ter119', 'CD235a']
    marker_list = ['CD4']
    
    # Colors for nuclei, cytoplasm and marker
    if show_eosin1 or show_eosin2:
        # Dark purple
        nuclei_color = np.array([72, 61, 139]) / 255.0  # Converted to 0-1 range
    else:
        # Blue
        nuclei_color = np.array([46, 77, 160]) / 255.0  # Converted to 0-1 range
    np.array([0, 77, 160]) / 255.0
    # Pink
    cytoplasm_color1 = np.array([255, 182, 193]) / 255.0  # Converted to 0-1 range

    # Purple
    cytoplasm_color2 = np.array([199, 143, 187]) / 255.0  # Converted to 0-1 range
    bloodcells_color = np.array([186, 56, 69]) / 255.0  # Converted to 0-1 range
    marker_color = np.array([180, 100, 0]) / 255.0  # Converted to 0-1 range
    
    
    
    with tifffile.TiffFile(input_filename) as tif:
        imc_image = tif.asarray()
        metadata = tif.pages[0].tags['ImageDescription'].value

    if imc_image.ndim == 3:
        imc_image = np.expand_dims(imc_image, axis=0)
        print(imc_image.shape)  # The shape will now be (1, height, width, channels)

    # print("Data size: ", imc_image.shape)
    imc_image = imc_image[0:1, ...]

    print("Data size: ", imc_image.shape)
    print("Image size: ", imc_image.shape[2:4])
    print("Number of channels: ", imc_image.shape[0])

    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    root = ET.fromstring(metadata)
    channel_elements = root.findall('.//ome:Channel', ns)
    channel_names = [channel.get('Name') for channel in channel_elements]

    print("Channel names: ", channel_names)
    
    
    
    
    
    if use_chatgpt:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        channel_names_string = ', '.join(channel_names)
        content = "Consider the following channels in a multiplexed image: " + channel_names_string + \
        " I want to convert this multiplexed image into a pseudo H&E image." + \
        " Which channels would be shown as blue, pink or purple in an H&E image, and which are shown as red (red blood cells)?" + \
        '''
        For simulating the pink appearance in a pseudo H&E image using multiplexed imaging like Imaging Mass Cytometry (IMC),
        focus on channels that tag proteins predominantly located in the cytoplasm and extracellular matrix.
        These are areas where eosin, which stains acidic components of the cell such as proteins, typically binds in traditional H&E staining.
        Proteins like collagen, which is a major component of the extracellular matrix, and fibronectin, another matrix protein,
        are ideal for this purpose. Additionally, cytoplasmic proteins such as cytokeratins in epithelial cells and muscle actin in muscle tissues would also appear pink,
        reflecting their substantial protein content and eosinophilic properties.

        For achieving a purple hue, the approach involves selecting channels that label proteins found both in the nucleus and in the cytoplasm,
        or at their interface. It includes markers associated with epithelial cells and other specific dense structures, giving a purple hue due to the density and nature of these proteins.
        This color is typically seen where there is a merging of the blue nuclear staining and the pink cytoplasmic staining.
        Intermediate filament proteins like cytokeratins, which are densely packed in the cytoplasm, and vimentin, common in mesenchymal cells, are key targets.
        Membrane proteins such as Caveolin-1, which is localized to the plasma membrane, can also contribute to this effect.
        These proteins, due to their strategic locations and the properties of the tagged antibodies used,
        allow for a nuanced blending of blue and pink, creating the purple appearance commonly observed in regions of cell-cell interaction or dense cytoplasmic content in traditional H&E staining.

        For highlighting red blood cells with vivid red, choosing the right markers is crucial.
        Ter119 is an ideal choice, as it specifically targets a protein expressed on the surface of erythroid cells in mice, from early progenitors to mature erythrocytes.
        This marker, when conjugated with a metal isotope, allows for precise visualization of red blood cells within tissue sections.
        To simulate the red appearance typical of eosin staining in traditional histology,
        Ter119 can be assigned a bright red color in the image analysis software.
        Additionally, targeting hemoglobin with a specific antibody can also serve as an alternative or complementary approach,
        as hemoglobin is abundant in red blood cells and can be visualized similarly by assigning a red color to its corresponding channel.
        Both strategies ensure that red blood cells stand out distinctly in the IMC data,
        providing a clear contrast to other cellular components and mimicking the traditional histological look.

        Here are some specific examples:

        Markers typically staining the cytoplasm, cytoskeletal elements, or ECM (Pink Eosin):
        Collagen (Col1A1, Col3A1, Col4A1): Collagens are major components of the ECM and appear pink in H&E staining.
        Fibronectin (FN): An ECM glycoprotein that helps in cell adhesion and migration, typically stained pink.
        Vimentin (VIM): An intermediate filament protein found in mesenchymal cells, contributes to the cytoplasmic structure, often stained pink.
        Alpha-smooth muscle actin (aSMA): Found in smooth muscle cells, it stains the cytoplasm and is often observed in connective tissue.
        CD31 (PECAM-1): Found on endothelial cells lining the blood vessels; staining can reveal the cytoplasmic extensions.
        Desmin: An intermediate filament in muscle cells, contributing to cytoplasmic staining.
        Laminin: A component of the basal lamina (part of the ECM), often appears pink in H&E staining.
        Actin: A cytoskeletal protein found throughout the cytoplasm.
        Fibronectin: Another ECM protein, contributing to the pink staining of connective tissues.

        Markers typically staining epithelial cells and other specific structures (Purple Eosin):
        Pan-cytokeratin (panCK), CK7, CK20: Cytokeratins are intermediate filament proteins in epithelial cells, and their dense networks can give a purple hue.
        Caveolin-1 (CAV1): Involved in caveolae formation in the cell membrane, often in epithelial and endothelial cells.
        Aquaporin-1 (AQP1): A water channel protein found in various epithelial and endothelial cells.
        EpCAM (EPCAM): An epithelial cell adhesion molecule, important in epithelial cell-cell adhesion.
        E-cadherin, P-cadherin: Adhesion molecules in epithelial cells, contributing to cell-cell junctions, often seen in purple.
        Mucin 1 (MUC1): A glycoprotein expressed on the surface of epithelial cells, contributing to the viscous secretions.
        S100: A protein often used to mark nerve cells, melanocytes, and others, contributing to more specific staining often appearing purple.
        SMA (Smooth Muscle Actin): Although sometimes considered under cytoskeletal, it often appears in more dense structures giving a purple hue.

        Markers specific to red blood cells:
        Ter119: A marker specific to erythroid cells (red blood cells).
        CD235a (Glycophorin A): Another marker specific to erythrocytes.

        ''' + \
        "Give a list of the channels as json file with \"blue\", \"plink\", \"purple\" and \"red\" as classes." + \
        "Double check your response to make sure it makes sense. Make sure the channels you give are also in the provided list above. A channel can not be in more than one group."

        completion = client.chat.completions.create(
            # model="gpt-4-turbo",
            model="gpt-4o-mini",
            messages=[
            {"role": "system", "content": "You are an expert in digital pathology, imaging mass cytometry, multiplexed imaging and spatial omics."},
            {"role": "user", "content": content}
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        # print(completion.choices[0].message.content)

        json_string = completion.choices[0].message.content

        # Parse JSON string to dictionary
        data = json.loads(json_string)

        # Extract lists for blue, pink, and purple
        blue_list = data['blue']
        pink_list = data['pink']
        purple_list = data['purple']
        red_list = data['red']
    else:
        blue_list = find_channels(channel_names, haematoxylin_list )
        pink_list = find_channels(channel_names, pink_eosin_list )
        purple_list = find_channels(channel_names, purple_eosin_list )
        red_list = find_channels(channel_names, blood_list )

    brown_list = find_channels(channel_names, marker_list )

    # Print lists to verify
    print("Blue channels:", blue_list)
    print("Pink channels:", pink_list)
    print("Purple channels:", purple_list)
    print("Red channels:", red_list)
    print("Brown channels:", brown_list)
    
    


    if blue_list:
        hematoxylin_image = np.maximum.reduce([
            normalize(imc_image[:, channel_names.index(ch), :, :], 1, 99) for ch in blue_list
        ])

        if blue_median_filter_size > 0:
            # hematoxylin_image = median(hematoxylin_image[i, ...], disk(blue_median_filter_size))
            hematoxylin_image = np.stack(
                [median(hematoxylin_image[i, ...], disk(blue_median_filter_size)) for i in range(hematoxylin_image.shape[0])],
                axis=0
            )

        if blue_gaussian_filter_sigma > 0:
            # hematoxylin_image = gaussian(hematoxylin_image, sigma=blue_gaussian_filter_sigma)
            hematoxylin_image = np.stack(
                [gaussian(hematoxylin_image[i, ...], sigma=blue_gaussian_filter_sigma) for i in range(hematoxylin_image.shape[0])],
                axis=0
            )

        # hematoxylin_image = normalize(hematoxylin_image, 1,99)
        # hematoxylin_image = np.clip(hematoxylin_image, 0, 1)

        hematoxylin_image = np.stack(
            [np.clip(normalize(hematoxylin_image[i, ...], 1, 99), 0, 1) for i in range(hematoxylin_image.shape[0])],
            axis=0
        )

        if use_haematoxylin_histogram_normalisation:
            kernel_size = (50, 50)  # you can also try different sizes or None
            clip_limit = 0.02       # adjust this to control contrast enhancement
            nbins = 256             # typically 256, but can be adjusted
            # Apply adaptive histogram equalization with parameters
            # hematoxylin_image = exposure.equalize_adapthist(hematoxylin_image, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)
            hematoxylin_image = np.stack(
                [
                    exposure.equalize_adapthist(
                        hematoxylin_image[i, ...], 
                        kernel_size=kernel_size, 
                        clip_limit=clip_limit, 
                        nbins=nbins
                    ) for i in range(hematoxylin_image.shape[0])
                ],
                axis=0
            )

    if pink_list:
        eosin_image1 = np.maximum.reduce([
            normalize(imc_image[:, channel_names.index(ch), :, :], 1, 99) for ch in pink_list
        ])

        if pink_median_filter_size > 0:
            # eosin_image1 = median(eosin_image1, disk(pink_median_filter_size))
            eosin_image1 = np.stack(
                [median(eosin_image1[i, ...], disk(pink_median_filter_size)) for i in range(eosin_image1.shape[0])],
                axis=0
            )
        if pink_gaussian_filter_sigma > 0:
            # eosin_image1 = gaussian(eosin_image1, sigma=pink_gaussian_filter_sigma)
            eosin_image1 = np.stack(
                [gaussian(eosin_image1[i, ...], sigma=pink_gaussian_filter_sigma) for i in range(eosin_image1.shape[0])],
                axis=0
            )

        # eosin_image1 = normalize(eosin_image1, 1,99)
        # eosin_image1 = np.clip(eosin_image1, 0, 1)
        eosin_image1 = np.stack(
            [np.clip(normalize(eosin_image1[i, ...], 1, 99), 0, 1) for i in range(eosin_image1.shape[0])],
            axis=0
        )

        if use_eosin_histogram_normalisation:
            kernel_size = (50, 50)  # you can also try different sizes or None
            clip_limit = 0.02       # adjust this to control contrast enhancement
            nbins = 256             # typically 256, but can be adjusted
            # Apply adaptive histogram equalization with parameters
            # eosin_image1 = exposure.equalize_adapthist(eosin_image1, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)
            eosin_image1 = np.stack(
                [
                    exposure.equalize_adapthist(
                        eosin_image1[i, ...], 
                        kernel_size=kernel_size, 
                        clip_limit=clip_limit, 
                        nbins=nbins
                    ) for i in range(eosin_image1.shape[0])
                ],
                axis=0
            )

    if purple_list:
        eosin_image2 = np.maximum.reduce([
            normalize(imc_image[:, channel_names.index(ch), :, :], 1, 99) for ch in purple_list
        ])

        if purple_median_filter_size > 0:
            # eosin_image2 = median(eosin_image2, disk(purple_median_filter_size))
            eosin_image2 = np.stack(
                [median(eosin_image2[i, ...], disk(purple_median_filter_size)) for i in range(eosin_image2.shape[0])],
                axis=0
            )
        if purple_gaussian_filter_sigma > 0:
            # eosin_image2 = gaussian(eosin_image2, sigma=purple_gaussian_filter_sigma)
            eosin_image2 = np.stack(
                [gaussian(eosin_image2[i, ...], sigma=purple_gaussian_filter_sigma) for i in range(eosin_image2.shape[0])],
                axis=0
            )

        # eosin_image2 = normalize(eosin_image2, 1,99)
        # eosin_image2 = np.clip(eosin_image2, 0, 1)
        eosin_image2 = np.stack(
            [np.clip(normalize(eosin_image2[i, ...], 1, 99), 0, 1) for i in range(eosin_image2.shape[0])],
            axis=0
        )

        if use_eosin_histogram_normalisation:
            kernel_size = (50, 50)  # you can also try different sizes or None
            clip_limit = 0.02       # adjust this to control contrast enhancement
            nbins = 256             # typically 256, but can be adjusted
            # Apply adaptive histogram equalization with parameters
            # eosin_image2 = exposure.equalize_adapthist(eosin_image2, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)
            eosin_image2 = np.stack(
                [
                    exposure.equalize_adapthist(
                        eosin_image2[i, ...], 
                        kernel_size=kernel_size, 
                        clip_limit=clip_limit, 
                        nbins=nbins
                    ) for i in range(eosin_image2.shape[0])
                ],
                axis=0
            )

    if red_list:
        bloodcells_image1 = np.maximum.reduce([
            normalize(imc_image[:, channel_names.index(ch), :, :], 1, 99) for ch in red_list
        ])
        # bloodcells_image1 = median(bloodcells_image1, disk(2))
        # bloodcells_image1 = uniform_filter(bloodcells_image1, size=3)
        # Function to create a normalized disk-shaped kernel
        # def create_circular_mean_kernel(radius):
        #     kernel = disk(radius)
        #     return kernel / kernel.sum()
        # circular_kernel = create_circular_mean_kernel(radius=1)
        # bloodcells_image1 = convolve(bloodcells_image1, circular_kernel)

        if red_median_filter_size > 0:
            # bloodcells_image1 = median(bloodcells_image1, disk(red_median_filter_size))
            bloodcells_image1 = np.stack(
                [median(bloodcells_image1[i, ...], disk(red_median_filter_size)) for i in range(bloodcells_image1.shape[0])],
                axis=0
            )
        if red_gaussian_filter_sigma > 0:
            # bloodcells_image1 = gaussian(bloodcells_image1, sigma=red_gaussian_filter_sigma)
            bloodcells_image1 = np.stack(
                [gaussian(bloodcells_image1[i, ...], sigma=red_gaussian_filter_sigma) for i in range(bloodcells_image1.shape[0])],
                axis=0
            )

        # bloodcells_image1 = normalize(bloodcells_image1, 1,99)
        # bloodcells_image1 = np.clip(bloodcells_image1, 0, 1)
        bloodcells_image1 = np.stack(
            [np.clip(normalize(bloodcells_image1[i, ...], 1, 99), 0, 1) for i in range(bloodcells_image1.shape[0])],
            axis=0
        )

    if brown_list:
        marker_image1 = np.maximum.reduce([
            normalize(imc_image[:, channel_names.index(ch), :, :], 1, 99) for ch in brown_list
        ])

        if brown_median_filter_size > 0:
            # marker_image1 = median(marker_image1, disk(brown_median_filter_size))
            marker_image1 = np.stack(
                [median(marker_image1[i, ...], disk(brown_median_filter_size)) for i in range(marker_image1.shape[0])],
                axis=0
            )
        if brown_gaussian_filter_sigma > 0:
            # marker_image1 = gaussian(marker_image1, sigma=brown_gaussian_filter_sigma)
            marker_image1 = np.stack(
                [gaussian(marker_image1[i, ...], sigma=brown_gaussian_filter_sigma) for i in range(marker_image1.shape[0])],
                axis=0
            )

        # marker_image1 = normalize(marker_image1, 1,99)
        # marker_image1 = np.clip(marker_image1, 0, 1)
        marker_image1 = np.stack(
            [np.clip(normalize(marker_image1[i, ...], 1, 99), 0, 1) for i in range(marker_image1.shape[0])],
            axis=0
        )


    # Create RGB images for each component
    white_image = np.ones(hematoxylin_image.shape + (3,), dtype=np.float32)  # White background
    base_image = np.ones(hematoxylin_image.shape + (3,), dtype=np.float32)  # White background

    # Apply the color based on the intensity
    if show_eosin1:
        if pink_list:
            for i in range(3):
                base_image[..., i] -= (white_image[..., i] - cytoplasm_color1[i]) * eosin_image1

    if show_eosin2:
        if purple_list:
            for i in range(3):
                base_image[..., i] -= (white_image[..., i] - cytoplasm_color2[i]) * eosin_image2

    if show_haematoxylin and blue_list:
        for i in range(3):
            base_image[..., i] -= (white_image[..., i] - nuclei_color[i]) * hematoxylin_image

    if show_marker and brown_list:
        for i in range(3):
            base_image[..., i] -= (white_image[..., i] - marker_color[i]) * marker_image1

    if show_blood and red_list:
        for i in range(3):
            base_image[..., i] -= (white_image[..., i] - bloodcells_color[i]) * bloodcells_image1


    # Ensure the pixel values remain within the valid range
    base_image = np.clip(base_image, 0, 1)

    base_image_uint8 = (base_image * 255).astype(np.uint8)
    
    
    
    if reference_filename:
        reference_image = io.imread(reference_filename)
        base_image_uint8 = np.squeeze(base_image_uint8, axis=0)
        base_image_uint8 = exposure.match_histograms(base_image_uint8, reference_image, channel_axis=-1)
        base_image_uint8 = np.expand_dims(base_image_uint8, axis=0)

    

    # Save the image   
    
    # Define pixel sizes and physical size in Z
    pixel_size_x = 2 * 0.56  # micron
    pixel_size_y = 2 * 0.56  # micron
    physical_size_z = 20  # micron

    base_image_uint8_transpose = np.transpose(base_image_uint8, (0, 3, 1, 2))

    # Write the OME-TIFF file
    write_ome_tiff(data=base_image_uint8_transpose,
                   output_filename=output_filename,
                   pixel_size_x=pixel_size_x,
                   pixel_size_y=pixel_size_y,
                   physical_size_z=physical_size_z,
                   Unit='µm',
                   imagej=False, 
                   create_pyramid=True,
                   compression='zlib')

    print("The OME-TIFF file has been successfully written.")