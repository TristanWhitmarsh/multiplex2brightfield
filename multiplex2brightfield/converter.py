import os
import re
import json
import numpy as np
import tifffile
from xml.etree import ElementTree as ET
from csbdeep.utils import normalize
from skimage.filters import gaussian, median
from skimage.morphology import disk
from skimage import exposure, io
from skimage.io import imsave
from keras.models import load_model
from PIL import Image
from numpy2ometiff import write_ome_tiff
import SimpleITK as sitk
from lxml import etree
import requests
from tqdm import tqdm

def download_model():
    url = 'https://path.to.your.storage/model.h5'
    model_path = 'path/to/save/model.h5'
    
    if not os.path.exists(model_path):
        print("Downloading model...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(model_path, 'wb') as f, tqdm(
                desc=model_path,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                bar.update(len(data))
                f.write(data)
    else:
        print("Model already downloaded.")
    return model_path

def find_channels(channel_names, marker_channels):
    """
    Finds and returns channel names that match any marker channel name patterns provided.

    Parameters:
    - channel_names (list): A list of strings representing all channel names in the image.
    - marker_channels (list): A list of strings representing the target marker names to search for.

    Returns:
    - list: A list of matching channel names from channel_names that match any of the patterns in marker_channels.
    """
    found_channels = []

    # Preprocess marker channels by cleaning and creating flexible regex patterns
    regex_patterns = []
    for marker in marker_channels:
        # Clean and create a pattern by removing non-alphanumeric characters and converting to lowercase
        cleaned_marker = re.sub(r'[^a-zA-Z0-9]', '', marker.lower())
        # Create a regex pattern to match any instance of the cleaned marker as a substring
        pattern = re.compile(cleaned_marker, re.IGNORECASE)
        regex_patterns.append(pattern)

    # Search through channel names to find any matches
    for name in channel_names:
        # Clean the channel name in the same way for consistent matching
        cleaned_name = re.sub(r'[^a-zA-Z0-9]', '', name.lower())

        # Check if any pattern matches the cleaned channel name
        if any(pattern.search(cleaned_name) for pattern in regex_patterns):
            found_channels.append(name)

    return found_channels


        
def resample_rgb_slices(image_array, input_pixel_size_x, input_pixel_size_y, output_pixel_size_x, output_pixel_size_y, interpolation=sitk.sitkLinear):
    """
    Resamples each Z slice in a ZYXC NumPy array based on the input and output pixel sizes,
    treating RGB channels as a multi-component image.

    Args:
    - image_array (np.array): Input array of shape (Z, Y, X, C) with dtype uint8.
    - input_pixel_size_x (float): The input pixel size in the x direction.
    - input_pixel_size_y (float): The input pixel size in the y direction.
    - output_pixel_size_x (float): The desired output pixel size in the x direction.
    - output_pixel_size_y (float): The desired output pixel size in the y direction.
    - interpolation (SimpleITK Interpolator): Interpolation method for resampling.
      Options include:
      - sitk.sitkLinear (default)
      - sitk.sitkNearestNeighbor
      - sitk.sitkBSpline
      - etc.

    Returns:
    - resampled_array (np.array): Resampled array of shape (Z, new_Y, new_X, C) with dtype uint8.
    """
    # Get the number of slices and channels
    num_slices, height, width, channels = image_array.shape
    assert channels == 3, "This function is designed for RGB images with 3 channels."
    resampled_slices = []

    # Loop over each Z slice
    for z in range(num_slices):
        # Extract the Z slice as an RGB image
        rgb_slice = image_array[z, :, :, :]

        # Convert the NumPy array to SimpleITK Image
        sitk_image = sitk.GetImageFromArray(rgb_slice, isVector=True)

        # Set the spacing (pixel size) for the input image
        sitk_image.SetSpacing((input_pixel_size_x, input_pixel_size_y))

        # Calculate the new size based on output pixel size
        size = sitk_image.GetSize()
        new_size = [
            int(size[0] * (input_pixel_size_x / output_pixel_size_x)),
            int(size[1] * (input_pixel_size_y / output_pixel_size_y))
        ]

        # Resample the image
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetOutputSpacing((output_pixel_size_x, output_pixel_size_y))
        resample_filter.SetSize(new_size)
        resample_filter.SetInterpolator(interpolation)  # Use the specified interpolation method
        resample_filter.SetDefaultPixelValue(0)

        resampled_sitk_image = resample_filter.Execute(sitk_image)

        # Convert back to NumPy array and append to list
        resampled_rgb_slice = sitk.GetArrayFromImage(resampled_sitk_image)
        resampled_slices.append(resampled_rgb_slice)

    # Stack all the resampled slices along the Z dimension
    resampled_array = np.stack(resampled_slices, axis=0)

    return resampled_array

def process_tile(tile, model):
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
    h, w, _ = image.shape
    processed_image = np.zeros((h, w, 3))
    
    for y in range(0, h - tile_size + 1, step_size):
        for x in range(0, w - tile_size + 1, step_size):
            # Extract tile
            tile = image[y:y+tile_size, x:x+tile_size]
            
            # Process tile
            processed_tile = process_tile(tile, model)
            
            # Extract center part
            center_y, center_x = tile_size // 4, tile_size // 4
            processed_center = processed_tile[center_y:center_y+step_size, center_x:center_x+step_size]
            
            # Place processed center into the result image
            processed_image[y+center_y:y+center_y+step_size, x+center_x:x+center_x+step_size] = processed_center
    
    return processed_image.astype(np.uint8)



def EnhanceBrightfield(input_image):
    # Load the model
    model = load_model('model.h5')

    pad_size=256
    
    padded_image = np.pad(input_image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

    processed_padded_image = process_image_with_tiling(padded_image, model)

    final_image = processed_padded_image[pad_size:-pad_size, pad_size:-pad_size]

    return final_image




def convert(input_filename, output_filename, reference_filename=[],
    show_haematoxylin = True,
    show_eosin1 = True,
    show_eosin2 = True,
    show_blood = True,
    show_marker = False,
    marker = None,
    use_chatgpt = True,
    use_gemini = False,
    use_claude = False,
    use_haematoxylin_histogram_normalisation = True,
    use_eosin_histogram_normalisation = True,
    histogram_matching = False,
    channel_names=[], 
    pixel_size_x=1, 
    pixel_size_y=1, 
    physical_size_z=1, 
    imagej=False, 
    create_pyramid=True, 
    compression='zlib', 
    Unit='µm', 
    downsample_count=4, 
    apply_filter = False,
    filter_settings=None,
    AI_enhancement=False,
    output_pixel_size_x = None,
    output_pixel_size_y = None,
    output_physical_size_z = None,
    api_key = ''):


    # Default filter settings if none are provided
    if apply_filter:
        print("Applying filter")
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
    
        
    haematoxylin_list = [
        'DNA', 'DAPI', 'hoechst', 'hoechst 33342', 'hoechst 2', 'hoechst stain', 'Iridium', 
        'Iridium-191', 'Iridium-193', 'Ir191', 'Ir193', 'Iridium_10331253Ir191Di', 
        'Iridium_10331254Ir193Di', 'H3', 'H4', 'H3K27me3', 'H3K9me3',
        # 'Histone', 'Histone_1261726In113Di', 'Histone_473968La139Di'
    ]

    pink_eosin_list = [
        'Col1A1', 'Col3A1', 'Col4A1', 'Col I', 'Col III', 'Col IV', 'Collagen I', 'Collagen III', 
        'Collagen IV', 'FN', 'Fibronectin', 'Fibrone', 'VIM', 'Vimentin', 'Vimenti', 'aSMA', 
        'SMA', 'smooth muscle actin', 'CD31', 'PECAM1', 'PECAM-1', 'Desmin', 'Laminin', 
        'Actin', 'eosin', 'stroma', 'Keratin'
    ]
    
    purple_eosin_list = [
        'panCK', 'panCyto', 'pancytokeratin', 'cytokeratin', 'Pan-Cytokeratin', 'CK7', 'CK20', 'CAV1', 'Caveolin-1', 'AQP1', 
        'Aquaporin-1', 'EPCAM', 'EpCAM', 'epithelial cell adhesion molecule', 
        'E-cadherin', 'P-cadherin', 'Cadherin', 'MUC1', 'S100', 'epithelium'
    ]
    
    blood_list = [
        'Ter119', 'Ter-119', 'Ter 119', 'CD235a', 'Glycophorin A', 'erythrocyte marker'
    ]
    
    if marker is None:
        marker_list = [
            'CD3',    # T-cell marker
            'CD4',    # Helper T-cell marker
            'CD8',    # Cytotoxic T-cell marker
            'CD20',   # B-cell marker
            'CD45',   # Pan-leukocyte marker
            'CD68',   # Macrophage marker
            'CD56',   # NK cell marker
            'CD57',   # NK cell subset marker
            'CD11b',  # Myeloid cell marker, monocytes/macrophages
            'CD11c',  # Dendritic cell marker, also on monocytes
            'CD163',  # M2 macrophage marker
            'CD38',   # Activation marker on B-cells, T-cells, and plasma cells
            'CD25',   # Activation marker on T-cells (IL-2 receptor)
            'CD44',   # Adhesion molecule, often used in stem cells and immune cells
            'CD62L',  # L-selectin, adhesion molecule on leukocytes
            'CD40',   # Activation marker on B-cells and APCs (Antigen Presenting Cells)
            'CD279',  # PD-1, checkpoint protein on T-cells
            'CD127',  # IL-7 receptor, used to mark memory T-cells
            'FOXP3',  # Regulatory T-cell marker (Tregs)
            'CD21',   # Follicular dendritic cell and mature B-cell marker
            'CD15',   # Granulocyte marker, especially neutrophils
            'CD138',  # Plasma cell marker
            'CD5',    # T-cell and some B-cell subset marker
            'CD30',   # Activation marker on B-cells and T-cells, often used in lymphoma
            'CD10',   # Marker for germinal center B-cells and some leukemias
            'CD23',   # Activated B-cell and dendritic cell marker
            'CD31',   # PECAM-1, endothelial cell marker (blood vessels)
            'CD34',   # Hematopoietic stem cell and endothelial progenitor marker
            'CD1a',   # Langerhans cells and cortical thymocyte marker
            'BCL2',   # Anti-apoptotic protein, often used in B-cells and tumors
            'Ki67',   # Proliferation marker (marks cells in the cell cycle)
            'p53',    # Tumor suppressor protein, often used in cancer studies
            'S100',   # Used for neural cells, dendritic cells, and melanocytes
            'E-cadherin', # Cell adhesion protein, used in epithelial and some cancer studies
            'PD-L1',  # Immune checkpoint ligand, often used in cancer and immune studies
            'MHCII',  # Major Histocompatibility Complex II, on antigen-presenting cells
            'CD14',   # Monocyte and macrophage marker
            'CD1c',   # Dendritic cell marker
            'CD138',  # Syndecan-1, often used to identify plasma cells
            'ARG1',   # Arginase-1, marker of M2 macrophages
            'GLUT1',  # Glucose transporter, often upregulated in tumors
            'Ly6G',   # Marker for neutrophils and granulocytes
            'Granzyme B', # Cytotoxic marker in NK cells and cytotoxic T-cells
            'F4/80',  # Macrophage marker commonly used in mouse studies
            'TCRγδ',  # Gamma delta T-cell receptor marker
            'CD209',  # DC-SIGN, dendritic cell marker
            'Lyve-1', # Lymphatic vessel marker
            'ICOS',   # Inducible T-cell co-stimulator, marker of activated T-cells
            'GATA3',  # Transcription factor, often used to identify Th2 cells and some epithelial cells
            'ER',     # Estrogen receptor, common in breast tissue studies
            'PR',     # Progesterone receptor, common in breast tissue studies
            'HER2',   # Human epidermal growth factor receptor 2, common in breast cancer studies
            'MUC1',   # Mucin-1, common in epithelial and cancer cells
        ]
    else:
        marker_list = [marker]
    

    
    
    
    
    
    # Load the TIFF file and get the metadata
    with tifffile.TiffFile(input_filename) as tif:
        ome_metadata = tif.ome_metadata
        imc_image = tif.asarray()
        metadata = tif.pages[0].tags['ImageDescription'].value

    # Parse XML metadata using lxml
    root = etree.fromstring(ome_metadata.encode('utf-8'))

    # Find the Pixels element using a wildcard for the namespace
    pixels = root.find('.//{*}Pixels')

    if pixels is not None:
        # Extracting the attributes
        input_pixel_size_x = float(pixels.get('PhysicalSizeX', 1))
        input_pixel_size_y = float(pixels.get('PhysicalSizeY', 1))
        input_physical_size_z = float(pixels.get('PhysicalSizeZ', 1))
    else:
        # Fallback if the <Pixels> element is not found in the XML
        input_pixel_size_x = 1
        input_pixel_size_y = 1
        input_physical_size_z = 1
    
       
    if not output_pixel_size_x:
        output_pixel_size_x = input_pixel_size_x
    if not output_pixel_size_y:
        output_pixel_size_y = input_pixel_size_y
    if not output_physical_size_z:
        output_physical_size_z = input_physical_size_z
    
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

    # print("Data size: ", imc_image.shape)
    imc_image = imc_image[0:1, ...]

    print("Data size: ", imc_image.shape)
    print("Image size: ", imc_image.shape[2:4])
    print("Number of channels: ", imc_image.shape[0])

    # Determine namespace based on the XML root's namespace
    ns_uri = root.tag.split('}')[0].strip('{')
    ns = {'ome': ns_uri}

    root = ET.fromstring(metadata)
    channel_elements = root.findall('.//ome:Channel', ns)
    channel_names = [channel.get('Name') for channel in channel_elements if channel.get('Name')]

    print("Channel names: ", channel_names)
    
    
    

    # Channel names provided
    channel_names_string = ', '.join(channel_names)
    
    content = "Consider the following channels in a multiplexed image: " + channel_names_string + \
        " I want to convert this multiplexed image into a pseudo H&E image." + \
        " Which channels would be shown as blue, pink or purple in an H&E image, and which are shown as red (red blood cells)?" + \
        '''
        
        Pink
        For simulating the pink appearance in a pseudo H&E image using multiplexed imaging like Imaging Mass Cytometry (IMC),
        focus on channels that tag proteins predominantly located in the cytoplasm and extracellular matrix.
        These are areas where eosin, which stains acidic components of the cell such as proteins, typically binds in traditional H&E staining.
        Proteins like collagen, which is a major component of the extracellular matrix, and fibronectin, another matrix protein,
        are ideal for this purpose. Additionally, cytoplasmic proteins such as cytokeratins in epithelial cells and muscle actin in muscle tissues would also appear pink,
        reflecting their substantial protein content and eosinophilic properties. It should not include markers that only stain the nucleus. Only include markers that predominantly stain the **cytoplasm or extracellular matrix (ECM)** and do not overlap significantly with nuclear components. This includes proteins like smooth muscle actin (SMA) and fibronectin, which are primarily found in the cytoplasm and cytoskeleton of cells without creating a dense, nuclear-interacting appearance.

        Purple:
        For achieving a purple hue, the approach involves selecting channels that label proteins found both in the nucleus and in the cytoplasm,
        or at their interface. It includes markers associated with epithelial cells and other specific dense structures, giving a purple hue due to the density and nature of these proteins.
        This color is typically seen where there is a merging of the blue nuclear staining and the pink cytoplasmic staining.
        Intermediate filament proteins like cytokeratins, which are densely packed in the cytoplasm, and vimentin, common in mesenchymal cells, are key targets.
        Membrane proteins such as Caveolin-1, which is localized to the plasma membrane, can also contribute to this effect.
        These proteins, due to their strategic locations and the properties of the tagged antibodies used,
        allow for a nuanced blending of blue and pink, creating the purple appearance commonly observed in regions of cell-cell interaction or dense cytoplasmic content in traditional H&E staining. It should not include markers that only stain the nucleus. Select markers found in **densely packed or epithelial cell structures** where there is a clear interaction between nuclear and cytoplasmic staining, creating a purple effect. Avoid including cytoplasmic-only proteins like SMA in this category, as they contribute to an overall pink appearance without significant nuclear overlap.

        Red:
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
        
        Markers for cell nuclei (Blue hematoxylin):
        DNA: this is a standard nuclear stain.
        DAPI: DAPI binds directly to DNA, staining all nuclei.
        Histone: histones are universal nuclear proteins, staining nuclei broadly.
        Iridium: iridium markers (e.g., Iridium_10331253Ir191Di, Iridium_10331254Ir193Di) intercalate with DNA, staining all nuclei.
        hoechst: Hoechst stains DNA universally in nuclei.

        Markers typically staining the cytoplasm, cytoskeletal elements, or ECM (Pink Eosin):
        Collagen (Col1A1, Col3A1, Col4A1): Collagens are major components of the ECM and appear pink in H&E staining.
        Fibronectin (FN): An ECM glycoprotein that helps in cell adhesion and migration, typically stained pink.
        Vimentin (VIM): An intermediate filament protein found in mesenchymal cells, contributes to the cytoplasmic structure, often stained pink.
        smooth muscle actin (aSMA, SMA): Found in smooth muscle cells, it stains the cytoplasm and is often observed in connective tissue.
        CD31 (PECAM-1): Found on endothelial cells lining the blood vessels; staining can reveal the cytoplasmic extensions.
        Desmin: An intermediate filament in muscle cells, contributing to cytoplasmic staining.
        Laminin: A component of the basal lamina (part of the ECM), often appears pink in H&E staining.
        Actin: A cytoskeletal protein found throughout the cytoplasm.
        CD68: a macrophage marker typically seen as cytoplasmic and eosinophilic
        Keratin: General keratins are cytoplasmic structural proteins in keratinized tissues like skin and hair, supporting cell structure.

        Markers typically staining epithelial cells and other specific structures (Purple Eosin):
        cytokeratin and Pan-Cytokeratin (panCK, CK7, CK20): Cytokeratins are intermediate filament proteins in epithelial cells, and their dense networks can give a purple hue.
        Caveolin-1 (CAV1): Involved in caveolae formation in the cell membrane, often in epithelial and endothelial cells.
        Aquaporin-1 (AQP1): A water channel protein found in various epithelial and endothelial cells.
        EpCAM (EPCAM): An epithelial cell adhesion molecule, important in epithelial cell-cell adhesion.
        E-cadherin, P-cadherin: Adhesion molecules in epithelial cells, contributing to cell-cell junctions, often seen in purple.
        Mucin 1 (MUC1): A glycoprotein expressed on the surface of epithelial cells, contributing to the viscous secretions.
        S100: A protein often used to mark nerve cells, melanocytes, and others, contributing to more specific staining often appearing purple.
        Lyve1: Lymphatic endothelial marker often found near epithelial structures.

        Markers specific to red blood cells:
        Ter119: A marker specific to erythroid cells (red blood cells).
        CD235a (Glycophorin A): Another marker specific to erythrocytes.
        
        Excluded markers:
        CD markers that do not appear in H&E: CD3, CD4, CD8, CD19, CD20, CD45, CD68, CD56, CD57, CD163, CD11c, CD38.
        cytokine receptors: PD-L1, IL18Ra, ICOS, CD40, CD25, CD62L, CD44, CD279 (PD-1).
        proliferation markers: Ki67, pH3 (phosphorylated Histone H3).
        Apoptosis and Stress Markers: Cleaved Caspase-3 (Casp3), BCL2, ASNS, ARG1, GLUT1.
        Phosphorylated Proteins: pS6, p53, pERK, pSTAT3.
        Oncoproteins and Tumor Markers:  c-Myc, EGFR, c-erbB, HER2, GATA3, Sox9.
        Isotopic Controls and Non-Biological Labels: Ruthenium and xenon isotopes.
        Mitochondrial and Organelle-Specific Markers: TOM20, ATP5A, VDAC1 (Voltage-Dependent Anion Channel)

        ''' + \
        "Give a list of the channels as json file with \"blue\", \"pink\", \"purple\" and \"red\" as classes." + \
        "Double check your response to make sure it makes sense. Make sure the channels you give are also in the provided list above. A channel can not be in more than one group. If you add cells for blood they must be specific for red blood cells and not just a nuclear stain that also stains red blood cells. Markers that bind directly to DNA or nucleic acid structures and thus stain all nuclei should be included in the blue category. Markers for phosphorylated proteins or signaling molecules should not be included, as they stain only cells where these specific proteins are expressed or active. When you make a list of channels you can only use channel names that are provided above. Forinstance, you can not just make up a name like CD31_77877Nd146Di."
    
    
    
    
    if use_chatgpt:
        from openai import OpenAI
        print("Using ChatGPT")
        client = OpenAI(api_key=api_key)
        
        completion = client.chat.completions.create(
            # model="gpt-4-turbo",
            model="gpt-4o",
            # model="gpt-4o-mini",
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
    elif use_gemini:
        print("Using Gemini")
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)

        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        response = model.generate_content(content)
        # print(response.text)
        
        # Check if the response contains valid text
        if response and response.text:
            try:
                # Remove markdown tags such as ```json
                cleaned_json_string = re.sub(r"```(?:json)?", "", response.text).strip()

                # Now you can safely parse the cleaned JSON string
                data = json.loads(cleaned_json_string)

                # Extract lists for blue, pink, purple, and red
                blue_list = data['blue']
                pink_list = data['pink']
                purple_list = data['purple']
                red_list = data['red']
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
    elif use_claude:
        print("Using Claude")
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        response = client.completions.create(
            model="claude-1.3",
            prompt=f"{anthropic.HUMAN_PROMPT} {content} {anthropic.AI_PROMPT}",
            max_tokens_to_sample=512,
            temperature=0,
        )
        
        print(response)

        # Check if the response contains valid text
        if response and response.completion:
            try:
                # Remove markdown tags such as ```json
                cleaned_json_string = re.sub(r"```(?:json)?", "", response.completion).strip()

                # Now you can safely parse the cleaned JSON string
                data = json.loads(cleaned_json_string)

                # Extract lists for blue, pink, purple, and red
                blue_list = data['blue']
                pink_list = data['pink']
                purple_list = data['purple']
                red_list = data['red']
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
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
    
    print()
    
    
    # Colors for nuclei, cytoplasm and marker
    if (show_eosin1 or show_eosin2) and (pink_list or purple_list):
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
    
    
    if output_pixel_size_x != input_pixel_size_x or output_pixel_size_y != input_pixel_size_y:
        print("Resampling")
        if AI_enhancement:
            interpolation = sitk.sitkNearestNeighbor
        else:
            interpolation = sitk.sitkLinear
        base_image_uint8 = resample_rgb_slices(base_image_uint8, input_pixel_size_x, input_pixel_size_y, output_pixel_size_x, output_pixel_size_y, interpolation=interpolation)

    
    if AI_enhancement:
        print("Enhancing image")
        base_image_uint8 = np.squeeze(base_image_uint8, axis=0)
        base_image_uint8 = EnhanceBrightfield(base_image_uint8)  
        base_image_uint8 = np.expand_dims(base_image_uint8, axis=0)
    
    if reference_filename:
        print("Appling histogram matching")
        reference_image = io.imread(reference_filename)
        base_image_uint8 = np.squeeze(base_image_uint8, axis=0)
        base_image_uint8 = exposure.match_histograms(base_image_uint8, reference_image, channel_axis=-1)
        base_image_uint8 = np.expand_dims(base_image_uint8, axis=0)


    base_image_uint8_transpose = np.transpose(base_image_uint8, (0, 3, 1, 2))

    # Write the OME-TIFF file
    write_ome_tiff(data=base_image_uint8_transpose,
                   output_filename=output_filename,
                   pixel_size_x=output_pixel_size_x,
                   pixel_size_y=output_pixel_size_y,
                   physical_size_z=output_physical_size_z,
                   Unit='µm',
                   imagej=False, 
                   create_pyramid=True,
                   compression='zlib')

    print("The OME-TIFF file has been successfully written.")