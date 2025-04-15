prompt = (
    " I want to convert this multiplexed image into a pseudo H&E image."
    + " Use the below list to see which channels would be shown as blue, pink, purple or red in an H&E image. Use your knowledge on spatial proteomics like IMC and other multiplexed imaging to understand what markers are in the multiplexed image and the markers below in the **BLUE**, **PINK**, **PURPLE** or **RED** list. Consider that markers/proteins can be written in many different ways. Do not use markers that are in the **EXCLUDE** list. "
    + """

        **BLUE** Markers for cell nuclei (Blue hematoxylin):
        1. DNA: this is a standard nuclear stain.
        2. DAPI: DAPI binds directly to DNA, staining all nuclei.
        3. Iridium: iridium markers (Iridium, Ir191, Ir193) intercalate with DNA, staining all nuclei.
        4. hoechst: Hoechst stains DNA universally in nuclei.
        5. Histone: histones are universal nuclear proteins, staining nuclei broadly. Only use Histone for the nucleir marker if no other markers for the cell nuclei are found.

        **PINK** Markers typically staining the cytoplasm, cytoskeletal elements, or ECM (Pink Eosin):
        1. Collagen (Collagen Type I, Col1A1, Col3A1, Col4A1, Col1): Collagens are major components of the ECM and appear pink in H&E staining.
        2. Fibronectin (FN): An ECM glycoprotein that helps in cell adhesion and migration, typically stained pink.
        3. Vimentin (VIM): An intermediate filament protein found in mesenchymal cells, contributes to the cytoplasmic structure, often stained pink.
        4. smooth muscle actin (aSMA, SMA): Found in smooth muscle cells, it stains the cytoplasm and is often observed in connective tissue.
        5. CD31 (PECAM-1, CD-31): Found on endothelial cells lining the blood vessels; staining can reveal the cytoplasmic extensions.
        6. CD34: Marks endothelial cells in blood vessels and certain fibroblasts
        7. Desmin: An intermediate filament in muscle cells, contributing to cytoplasmic staining.
        8. Laminin: A component of the basal lamina (part of the ECM), often appears pink in H&E staining.
        9. Actin: A cytoskeletal protein found throughout the cytoplasm.
        10. Keratin: General keratins are cytoplasmic structural proteins in keratinized tissues like skin and hair, supporting cell structure.
        11. CD39: It is found on the surface of endothelial cells, making it useful for visualizing blood vessels and vascular structures in tissue.
        12. CD56: It is expressed in muscle tissues, and muscle is typically stained pink by eosin in H&E.
        13. CD90/Thy1: a glycoprotein found on the surface of various cell types, including fibroblasts, endothelial cells, neurons, and T cells.
        14. AF555 (Alexa Fluor 555): This fluorophore is frequently conjugated with antibodies targeting cytoplasmic proteins or elements within the extracellular matrix (ECM)
        15. AF488 (Alexa Fluor 488): AF488 is often used to label cytoplasmic and ECM proteins
        16. Lyve1: Lymphatic endothelial marker

        **PURPLE** Markers typically staining epithelial cells and other specific structures (Purple Eosin):
        1. cytokeratin and Pan-Cytokeratin (panCK, CK7, CK20, Pan-CK): Cytokeratins are intermediate filament proteins in epithelial cells, and their dense networks can give a purple hue.
        2. Caveolin-1 (CAV1): Involved in caveolae formation in the cell membrane, often in epithelial and endothelial cells.
        3. Aquaporin-1 (AQP1): A water channel protein found in various epithelial and endothelial cells.
        4. EpCAM (EPCAM): An epithelial cell adhesion molecule, important in epithelial cell-cell adhesion.
        5. E-cadherin (E-cad), P-cadherin (P-cad): Adhesion molecules in epithelial cells, contributing to cell-cell junctions, often seen in purple.
        6. Mucin 1 (MUC1): A glycoprotein expressed on the surface of epithelial cells, contributing to the viscous secretions.
        7. S100: A protein often used to mark nerve cells, melanocytes, and others, contributing to more specific staining often appearing purple.
        8. CD40: Given its distribution in lymphocyte-rich and immune-interactive regions, CD40 would be suitable for the purple group in a virtual H&E image.
        9. Podoplanin (PDPN): Another marker for lymphatic endothelium, often used alongside LYVE-1.
        10. Prox1: A transcription factor critical for lymphatic endothelial cell identity, sometimes used in combination with LYVE-1 for lymphatic vessel identification.
        11. Cy5 is typically used to label proteins in denser, structural contexts or epithelial markers

        **RED** Markers specific to red blood cells:
        1. Ter119: A marker specific to erythroid cells (red blood cells).
        2. CD235a (Glycophorin A): Another marker specific to erythrocytes.

        **EXCLUDE** Excluded markers:
        1. CD markers that do not appear in H&E: CD2 (T11), CD3, CD4, CD6 (T12), CD8, CD19, CD20, CD45, CD68, CD56, CD57, CD163, CD11c, CD38 (T10).
        2. cytokine receptors: PD-L1, IL18Ra, ICOS, CD40, CD25, CD62L, CD44, CD279 (PD-1).
        3. proliferation markers: Ki67, pH3 (phosphorylated Histone H3).
        4. Apoptosis and Stress Markers: Cleaved Caspase-3 (Casp3), BCL2, ASNS, ARG1, GLUT1.
        5. Phosphorylated Proteins: pS6, p53, pERK, pSTAT3.
        6. Oncoproteins and Tumor Markers:  c-Myc, EGFR, c-erbB, HER2, GATA3, Sox9.
        7. Isotopic Controls and Non-Biological Labels: Ruthenium and xenon isotopes.
        8. Mitochondrial and Organelle-Specific Markers: TOM20, ATP5A, VDAC1 (Voltage-Dependent Anion Channel)
        9. CD68: a macrophage marker typically seen as cytoplasmic and eosinophilic


        If a marker in the multiplexed image could not be found in these lists, use the below descriptions to see if they still belong in the **Blue**, **Pink**, **Purple** or **Red** list. Exclude markers that don't belong in any of them.

        **Blue** All markers that highlight the cell nucleus. These don not include markers that only highlight a few cell nuclei, they must highlight all the cell nuclei.

        **Pink** For simulating the pink appearance in a pseudo H&E image using multiplexed imaging like Imaging Mass Cytometry (IMC),
        focus on channels that tag proteins predominantly located in the cytoplasm and extracellular matrix.
        These are areas where eosin, which stains acidic components of the cell such as proteins, typically binds in traditional H&E staining.
        Proteins like collagen, which is a major component of the extracellular matrix, and fibronectin, another matrix protein,
        are ideal for this purpose. Additionally, cytoplasmic proteins such as cytokeratins in epithelial cells and muscle actin in muscle tissues would also appear pink,
        reflecting their substantial protein content and eosinophilic properties. It should not include markers that only stain the nucleus. Only include markers that predominantly stain the **cytoplasm or extracellular matrix (ECM)** and do not overlap significantly with nuclear components. This includes proteins like smooth muscle actin (SMA) and fibronectin, which are primarily found in the cytoplasm and cytoskeleton of cells without creating a dense, nuclear-interacting appearance.

        **Purple** For achieving a purple hue, the approach involves selecting channels that label proteins found both in the nucleus and in the cytoplasm,
        or at their interface. It includes markers associated with epithelial cells and other specific dense structures, giving a purple hue due to the density and nature of these proteins.
        This color is typically seen where there is a merging of the blue nuclear staining and the pink cytoplasmic staining.
        Intermediate filament proteins like cytokeratins, which are densely packed in the cytoplasm, and vimentin, common in mesenchymal cells, are key targets.
        Membrane proteins such as Caveolin-1, which is localized to the plasma membrane, can also contribute to this effect.
        These proteins, due to their strategic locations and the properties of the tagged antibodies used,
        allow for a nuanced blending of blue and pink, creating the purple appearance commonly observed in regions of cell-cell interaction or dense cytoplasmic content in traditional H&E staining. It should not include markers that only stain the nucleus. Select markers found in **densely packed or epithelial cell structures** where there is a clear interaction between nuclear and cytoplasmic staining, creating a purple effect. Avoid including cytoplasmic-only proteins like SMA in this category, as they contribute to an overall pink appearance without significant nuclear overlap.

        **Red** For highlighting red blood cells with vivid red, choosing the right markers is crucial.
        Ter119 is an ideal choice, as it specifically targets a protein expressed on the surface of erythroid cells in mice, from early progenitors to mature erythrocytes.
        This marker, when conjugated with a metal isotope, allows for precise visualization of red blood cells within tissue sections.
        To simulate the red appearance typical of eosin staining in traditional histology,
        Ter119 can be assigned a bright red color in the image analysis software.
        Additionally, targeting hemoglobin with a specific antibody can also serve as an alternative or complementary approach,
        as hemoglobin is abundant in red blood cells and can be visualized similarly by assigning a red color to its corresponding channel.
        Both strategies ensure that red blood cells stand out distinctly in the IMC data,
        providing a clear contrast to other cellular components and mimicking the traditional histological look.

        """
    + 'Give a list of the channels as json file with "blue", "pink", "purple" and "red" as classes.'
    + "Double check your response to make sure it makes sense. Make sure the channels you give are also in the provided list above. A channel can not be in more than one group. If you add cells for blood they must be specific for red blood cells and not just a nuclear stain that also stains red blood cells. Markers that bind directly to DNA or nucleic acid structures and thus stain all nuclei should be included in the blue category. Markers for phosphorylated proteins or signaling molecules should not be included, as they stain only cells where these specific proteins are expressed or active. When you make a list of channels you can only use channel names that are provided above. Forinstance, you can not just make up a name like CD31_77877Nd146Di."
)


def GetConfiguration(name=None):
    """
    Retrieve the stain configuration dictionary for converting multiplex images to a virtual brightfield image.

    This function returns a configuration dictionary that defines the parameters and processing settings for
    stain simulation. The configuration includes properties for various stain components such as haematoxylin,
    eosinophilic, epithelial, erythrocytes, and marker channels, as well as the background color. Each component
    contains settings for color, intensity, filtering (median, Gaussian, and sharpening), histogram normalization,
    and channel targeting.

    By default, the configuration for H&E (Hematoxylin & Eosin) staining is returned. If a specific stain name is
    provided (for example, "IHC", "Aperio CS2", etc.), the function adjusts the configuration parameters accordingly.
    This allows flexible switching between different preset staining protocols.

    Args:
        name (str, optional): The name of the stain configuration to retrieve. If None, the default H&E configuration
                              is returned.

    Returns:
        dict: A dictionary containing the configuration parameters for the selected stain, including details for each
              component and the background.
    """
    
    config = {
        "name": "H&E",
        "components": {
            "haematoxylin": {
                "color": {
                    "R": 72,
                    "G": 61,
                    "B": 139,
                },
                "intensity": 1.0,
                "median_filter_size": 1,
                "gaussian_filter_sigma": 0,
                "sharpen_filter_radius": 2,
                "sharpen_filter_amount": 0,
                "histogram_normalisation": True,
                "normalize_percentage_min": 10,
                "normalize_percentage_max": 90,
                "clip": None,
                "targets": [
                    "DNA",
                    "DAPI",
                    "hoechst",
                    "hoechst 33342",
                    "hoechst 2",
                    "hoechst stain",
                    "Iridium",
                    "Iridium-191",
                    "Iridium-193",
                    "Ir191",
                    "Ir193",
                    "Iridium_10331254Ir193Di",
                    "H3",
                    "H4",
                    "H3K27me3",
                    "H3K9me3",
                ],
            },
            "eosinophilic": {
                "color": {
                    "R": 255,
                    "G": 182,
                    "B": 193,
                },
                "intensity": 1.0,
                "median_filter_size": 2,
                "gaussian_filter_sigma": 0.5,
                "sharpen_filter_radius": 2,
                "sharpen_filter_amount": 0,
                "histogram_normalisation": False,
                "normalize_percentage_min": 10,
                "normalize_percentage_max": 90,
                "clip": None,
                "targets": [
                    "Col",
                    "FN",
                    "Fibronectin",
                    "Fibrone",
                    "VIM",
                    "Vimentin",
                    "Vimenti",
                    "aSMA",
                    "SMA",
                    "smooth muscle actin",
                    "CD31",
                    "PECAM1",
                    "PECAM-1",
                    "Desmin",
                    "Laminin",
                    "Actin",
                    "eosin",
                    "stroma",
                    "Keratin",
                    "CD34",
                    "CD39",
                    "CD56",
                    "CD90",
                    "Thy1",
                    "AlexaFluor555",
                    "Alexa-Fluor-555",
                    "LYVE-1",
                    "LYVE",
                    "LYVE1",
                    "AlexaFluor488",
                    "Alexa-Fluor-488",
                    "muscle",
                    "CD66b",
                    "CD16",
                    "144Nd_pro-SPC",
                ],
            },
            "epithelial": {
                "color": {
                    "R": 199,
                    "G": 143,
                    "B": 187,
                },
                "intensity": 1.0,
                "median_filter_size": 2,
                "gaussian_filter_sigma": 1,
                "sharpen_filter_radius": 2,
                "sharpen_filter_amount": 0,
                "histogram_normalisation": False,
                "normalize_percentage_min": 10,
                "normalize_percentage_max": 90,
                "clip": None,
                "targets": [
                    "panCK",
                    "panCyto",
                    "pancytokeratin",
                    "cytokeratin",
                    "Pan-Cytokeratin",
                    "CK7",
                    "CK20",
                    "CAV1",
                    "Caveolin-1",
                    "Caveolin",
                    "AQP1",
                    "Aquaporin-1",
                    "Aquaporin",
                    "EPCAM",
                    "epithelial cell adhesion molecule",
                    "E-cadherin",
                    "P-cadherin",
                    "Cadherin",
                    "MUC1",
                    "S100",
                    "epithelium",
                    "S100",
                    "CD40",
                    "PDPN",
                    "Prox1",
                    "E-cadherin",
                    "E-cad",
                    "P-cadherin",
                    "P-cad",
                    "CDX-2",
                ],
            },
            "erythrocytes": {
                "color": {
                    "R": 186,
                    "G": 56,
                    "B": 69,
                },
                "intensity": 1.0,
                "median_filter_size": 1,
                "gaussian_filter_sigma": 0,
                "sharpen_filter_radius": 2,
                "sharpen_filter_amount": 0,
                "histogram_normalisation": False,
                "normalize_percentage_min": 10,
                "normalize_percentage_max": 90,
                "clip": None,
                "targets": [
                    "Ter119",
                    "Ter-119",
                    "Ter 119",
                    "CD235a",
                    "Glycophorin A",
                    "erythrocyte marker",
                    "AF",
                    "Autofluorescence",
                ],
            },
            "marker": {
                "color": {
                    "R": 180,
                    "G": 100,
                    "B": 0,
                },
                "intensity": 1.0,
                "median_filter_size": 2,
                "gaussian_filter_sigma": 1,
                "sharpen_filter_radius": 2,
                "sharpen_filter_amount": 0,
                "histogram_normalisation": False,
                "normalize_percentage_min": 10,
                "normalize_percentage_max": 90,
                "clip": None,
            },
        },
        "background": {
            "color": {
                "R": 255,
                "G": 255,
                "B": 255,
            },
        },
    }

    if name == "IHC":
        config["name"] = "IHC"
        config["components"]["haematoxylin"]["color"].update(
            {"R": 46, "G": 77, "B": 160}
        )
        del config["components"]["eosinophilic"]
        del config["components"]["epithelial"]
        del config["components"]["erythrocytes"]
        config["components"]["marker"]["targets"] = [
            "CD3",  # T-cell marker
            "CD4",  # Helper T-cell marker
            "CD8",  # Cytotoxic T-cell marker
            "CD20",  # B-cell marker
            "CD45",  # Pan-leukocyte marker
            "CD68",  # Macrophage marker
            "CD56",  # NK cell marker
            "CD57",  # NK cell subset marker
            "CD11b",  # Myeloid cell marker, monocytes/macrophages
            "CD11c",  # Dendritic cell marker, also on monocytes
            "CD163",  # M2 macrophage marker
            "CD38",  # Activation marker on B-cells, T-cells, and plasma cells
            "CD25",  # Activation marker on T-cells (IL-2 receptor)
            "CD44",  # Adhesion molecule, often used in stem cells and immune cells
            "CD62L",  # L-selectin, adhesion molecule on leukocytes
            "CD40",  # Activation marker on B-cells and APCs (Antigen Presenting Cells)
            "CD279",  # PD-1, checkpoint protein on T-cells
            "PD-1",  # CD279, checkpoint protein on T-cells
            "CD127",  # IL-7 receptor, used to mark memory T-cells
            "FOXP3",  # Regulatory T-cell marker (Tregs)
            "CD21",  # Follicular dendritic cell and mature B-cell marker
            "CD15",  # Granulocyte marker, especially neutrophils
            "CD138",  # Plasma cell marker
            "CD5",  # T-cell and some B-cell subset marker
            "CD30",  # Activation marker on B-cells and T-cells, often used in lymphoma
            "CD10",  # Marker for germinal center B-cells and some leukemias
            "CD23",  # Activated B-cell and dendritic cell marker
            "CD31",  # PECAM-1, endothelial cell marker (blood vessels)
            "CD34",  # Hematopoietic stem cell and endothelial progenitor marker
            "CD1a",  # Langerhans cells and cortical thymocyte marker
            "BCL2",  # Anti-apoptotic protein, often used in B-cells and tumors
            "Ki67",  # Proliferation marker (marks cells in the cell cycle)
            "p53",  # Tumor suppressor protein, often used in cancer studies
            "S100",  # Used for neural cells, dendritic cells, and melanocytes
            "E-cadherin",  # Cell adhesion protein, used in epithelial and some cancer studies
            "PD-L1",  # Immune checkpoint ligand, often used in cancer and immune studies
            "MHCII",  # Major Histocompatibility Complex II, on antigen-presenting cells
            "CD14",  # Monocyte and macrophage marker
            "CD1c",  # Dendritic cell marker
            "CD138",  # Syndecan-1, often used to identify plasma cells
            "ARG1",  # Arginase-1, marker of M2 macrophages
            "GLUT1",  # Glucose transporter, often upregulated in tumors
            "Ly6G",  # Marker for neutrophils and granulocytes
            "Granzyme B",  # Cytotoxic marker in NK cells and cytotoxic T-cells
            "F4/80",  # Macrophage marker commonly used in mouse studies
            "TCRγδ",  # Gamma delta T-cell receptor marker
            "CD209",  # DC-SIGN, dendritic cell marker
            "Lyve-1",  # Lymphatic vessel marker
            "ICOS",  # Inducible T-cell co-stimulator, marker of activated T-cells
            "GATA3",  # Transcription factor, often used to identify Th2 cells and some epithelial cells
            "ER",  # Estrogen receptor, common in breast tissue studies
            "PR",  # Progesterone receptor, common in breast tissue studies
            "HER2",  # Human epidermal growth factor receptor 2, common in breast cancer studies
            "MUC1",  # Mucin-1, common in epithelial and cancer cells
            "HLADR",  # MHC class II marker on antigen-presenting cells.
            "CD45RA",  # Marker of naïve T cells.
            "CD161",  # Marker of NK cells and select T cell subsets.
            "CD16",  # Fc receptor marker mediating ADCC on NK cells, neutrophils, and monocytes.
            "TCRVa",  # Marker for the variable region of the T-cell receptor alpha chain.
            "CD19",  # Pan-B cell marker.
            "CD66b",  # Neutrophil activation marker.
            "CD45RO",  # Marker of memory T cells.
            "CD69",  # Early activation marker on lymphocytes.
            "IL18Ra",  # IL-18 receptor alpha marker on T and NK cells.
        ]

    if name == "Aperio CS2":
        config["name"] = "Aperio CS2"
        config["components"]["haematoxylin"]["color"].update(
            {"R": 129, "G": 74, "B": 122}
        )
        config["components"]["eosinophilic"]["color"].update(
            {"R": 214, "G": 108, "B": 136}
        )
        config["components"]["epithelial"]["color"].update(
            {"R": 173, "G": 109, "B": 151}
        )
        config["components"]["erythrocytes"]["color"].update(
            {"R": 227, "G": 154, "B": 173}
        )
        config["components"]["marker"]["color"].update({"R": 126, "G": 50, "B": 47})
        config["background"]["color"].update({"R": 233, "G": 224, "B": 227})

    elif name == "Hamamatsu XR":
        config["name"] = "Hamamatsu XR"
        config["components"]["haematoxylin"]["color"].update(
            {"R": 125, "G": 72, "B": 166}
        )
        config["components"]["eosinophilic"]["color"].update(
            {"R": 222, "G": 101, "B": 191}
        )
        config["components"]["epithelial"]["color"].update(
            {"R": 208, "G": 151, "B": 211}
        )
        config["components"]["erythrocytes"]["color"].update(
            {"R": 203, "G": 62, "B": 150}
        )
        config["components"]["marker"]["color"].update({"R": 94, "G": 52, "B": 70})
        config["background"]["color"].update({"R": 224, "G": 216, "B": 227})

    elif name == "Hamamatsu S360":
        config["name"] = "Hamamatsu S360"
        config["components"]["haematoxylin"]["color"].update(
            {"R": 110, "G": 44, "B": 116}
        )
        config["components"]["eosinophilic"]["color"].update(
            {"R": 234, "G": 120, "B": 156}
        )
        config["components"]["epithelial"]["color"].update(
            {"R": 220, "G": 161, "B": 200}
        )
        config["components"]["erythrocytes"]["color"].update(
            {"R": 234, "G": 88, "B": 117}
        )
        config["components"]["marker"]["color"].update({"R": 121, "G": 49, "B": 59})
        config["background"]["color"].update({"R": 230, "G": 227, "B": 236})

    elif name == "Leica GT450":
        config["name"] = "Leica GT450"
        config["components"]["haematoxylin"]["color"].update(
            {"R": 101, "G": 48, "B": 151}
        )
        config["components"]["eosinophilic"]["color"].update(
            {"R": 247, "G": 152, "B": 205}
        )
        config["components"]["epithelial"]["color"].update(
            {"R": 236, "G": 178, "B": 228}
        )
        config["components"]["erythrocytes"]["color"].update(
            {"R": 254, "G": 153, "B": 209}
        )
        config["components"]["marker"]["color"].update({"R": 180, "G": 100, "B": 0})
        config["background"]["color"].update({"R": 251, "G": 245, "B": 251})

    elif name == "3DHistech Pannoramic Scan II":
        config["name"] = "3DHistech Pannoramic Scan II"
        config["components"]["haematoxylin"]["color"].update(
            {"R": 76, "G": 26, "B": 106}
        )
        config["components"]["eosinophilic"]["color"].update(
            {"R": 204, "G": 64, "B": 180}
        )
        config["components"]["epithelial"]["color"].update(
            {"R": 171, "G": 93, "B": 186}
        )
        config["components"]["erythrocytes"]["color"].update(
            {"R": 221, "G": 71, "B": 169}
        )
        config["components"]["marker"]["color"].update({"R": 166, "G": 70, "B": 58})
        config["background"]["color"].update({"R": 253, "G": 248, "B": 254})

    elif name == "CyteFinder":
        config["name"] = "CyteFinder"
        config["components"]["haematoxylin"]["color"].update(
            {"R": 37, "G": 28, "B": 111}
        )
        config["components"]["eosinophilic"]["color"].update(
            {"R": 213, "G": 37, "B": 162}
        )
        config["components"]["epithelial"]["color"].update(
            {"R": 144, "G": 102, "B": 171}
        )
        config["components"]["erythrocytes"]["color"].update(
            {"R": 164, "G": 49, "B": 96}
        )
        config["components"]["marker"]["color"].update({"R": 166, "G": 70, "B": 58})
        config["background"]["color"].update({"R": 253, "G": 248, "B": 254})

    elif name == "Axioscan 7":
        config["name"] = "Axioscan 7"
        config["components"]["haematoxylin"]["color"].update(
            {"R": 0, "G": 0, "B": 142, "intensity": 3.0}
        )
        config["components"]["eosinophilic"]["color"].update(
            {"R": 204, "G": 142, "B": 179, "intensity": 0.5}
        )
        config["components"]["epithelial"]["color"].update(
            {"R": 250, "G": 126, "B": 190, "intensity": 0.5}
        )
        config["components"]["erythrocytes"]["color"].update(
            {"R": 185, "G": 97, "B": 122}
        )
        config["components"]["marker"]["color"].update({"R": 45, "G": 15, "B": 17})
        config["background"]["color"].update({"R": 203, "G": 198, "B": 202})

    elif name == "Orion":
        config["name"] = "Orion"
        config["components"]["haematoxylin"]["color"].update(
            {"R": 0, "G": 0, "B": 126, "intensity": 0.5}
        )
        config["components"]["eosinophilic"]["color"].update(
            {"R": 213, "G": 181, "B": 205, "intensity": 3.0}
        )
        config["components"]["epithelial"]["color"].update(
            {"R": 219, "G": 211, "B": 223, "intensity": 2.0}
        )
        config["components"]["erythrocytes"]["color"].update(
            {"R": 221, "G": 152, "B": 180}
        )
        config["components"]["marker"]["color"].update({"R": 106, "G": 58, "B": 51})
        config["background"]["color"].update({"R": 249, "G": 249, "B": 249})

    elif name == "Masson Trichrome":
        config = {
            "name": "Masson Trichrome",
            "components": {
                "nuclei": {
                    "color": {
                        "R": 30,
                        "G": 114,
                        "B": 201,
                    },
                    "intensity": 0.5,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": [
                        "DNA",
                        "DAPI",
                        "hoechst",
                        "hoechst 33342",
                        "hoechst 2",
                        "hoechst stain",
                        "Iridium",
                        "Iridium-191",
                        "Iridium-193",
                        "Ir191",
                        "Ir193",
                        "Iridium_10331254Ir193Di",
                        "H3",
                        "H4",
                        "H3K27me3",
                        "H3K9me3",
                    ],
                },
                "muscle": {
                    "color": {
                        "R": 220,
                        "G": 67,
                        "B": 51,
                    },
                    "intensity": 1.0,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": ["141Pr_aSMA", "143Nd_VIM", "174Yb_Desmin"],
                },
                "collagen": {
                    "color": {
                        "R": 93,
                        "G": 209,
                        "B": 225,
                    },
                    "intensity": 1.0,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": ["146Nd_Col1A1", "150Nd_FN"],
                },
                "erythrocytes": {
                    "color": {
                        "R": 229,
                        "G": 128,
                        "B": 56,
                    },
                    "intensity": 0.5,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": [
                        "Ter119",
                        "Ter-119",
                        "Ter 119",
                        "CD235a",
                        "Glycophorin A",
                        "erythrocyte marker",
                    ],
                },
            },
            "background": {
                "color": {
                    "R": 255,
                    "G": 255,
                    "B": 255,
                },
            },
        }

    elif name == "PAS":
        config = {
            "name": "PAS",
            "components": {
                "nuclei": {
                    "color": {
                        "R": 50,
                        "G": 84,
                        "B": 210,
                    },
                    "intensity": 1.0,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": [
                        "DNA",
                        "DAPI",
                        "hoechst",
                        "hoechst 33342",
                        "hoechst 2",
                        "hoechst stain",
                        "Iridium",
                        "Iridium-191",
                        "Iridium-193",
                        "Ir191",
                        "Ir193",
                        "Iridium_10331254Ir193Di",
                        "H3",
                        "H4",
                        "H3K27me3",
                        "H3K9me3",
                    ],
                },
                "polysaccharides": {
                    "color": {
                        "R": 180,
                        "G": 80,
                        "B": 208,
                    },
                    "intensity": 1.0,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": ["144Nd_pro-SPC", "146Nd_Col1A1", "150Nd_FN"],
                },
                "stroma": {
                    "color": {
                        "R": 227,
                        "G": 186,
                        "B": 225,
                    },
                    "intensity": 1,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": [
                        "141Pr_aSMA",
                        "142Nd_CAV1",
                        "143Nd_VIM",
                        "164Dy_PDPN",
                        "174Yb_Desmin",
                    ],
                },
            },
            "background": {
                "color": {
                    "R": 250,
                    "G": 251,
                    "B": 255,
                },
            },
        }

    elif name == "Jones Silver":
        config = {
            "name": "Jones Silver",
            "components": {
                "membranes": {
                    "color": {
                        "R": 0,
                        "G": 0,
                        "B": 0,
                    },
                    "intensity": 1.5,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": ["150Nd_FN", "146Nd_Col1A1"],
                },
                "stroma": {
                    "color": {
                        "R": 133,
                        "G": 227,
                        "B": 200,
                    },
                    "intensity": 1.0,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": [
                        "141Pr_aSMA",
                        "143Nd_VIM",
                    ],
                },
            },
            "background": {
                "color": {
                    "R": 251,
                    "G": 245,
                    "B": 251,
                },
            },
        }

    elif name == "Toluidine Blue":
        config = {
            "name": "Toluidine Blue",
            "components": {
                "nuclei": {
                    "color": {
                        "R": 14,
                        "G": 21,
                        "B": 198,
                    },
                    "intensity": 0.5,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": [
                        "191Ir_DNA",
                        "193Ir_DNA",
                    ],
                },
                "stroma": {
                    "color": {
                        "R": 16,
                        "G": 193,
                        "B": 251,
                    },
                    "intensity": 1.0,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": [
                        "141Pr_aSMA",
                        "143Nd_VIM",
                        "146Nd_Col1A1",
                        "150Nd_FN",
                        "174Yb_Desmin",
                    ],
                },
                "metachromasia": {
                    "color": {
                        "R": 167,
                        "G": 154,
                        "B": 254,
                    },
                    "intensity": 3.0,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": ["144Nd_pro-SPC", "164Dy_PDPN"],
                },
            },
            "background": {
                "color": {
                    "R": 254,
                    "G": 255,
                    "B": 253,
                },
            },
        }

    elif name == "H&E ChatGPT":
        config = {
            "name": "H&E ChatGPT",
            "components": {
                "nuclei": {
                    "color": {
                        "R": 72,
                        "G": 61,
                        "B": 139,
                    },
                    "intensity": 1.0,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": ["191Ir_DNA", "193Ir_DNA"],
                },
                "eosinophilic": {
                    "color": {
                        "R": 255,
                        "G": 182,
                        "B": 193,
                    },
                    "intensity": 1.0,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": [
                        "141Pr_aSMA",
                        "143Nd_VIM",
                        "144Nd_pro-SPC",
                        "146Nd_Col1A1",
                        "150Nd_FN",
                        "162Dy_CD31",
                        "169Tm_LYVE1",
                        "174Yb_Desmin",
                    ],
                },
                "epithelial": {
                    "color": {
                        "R": 199,
                        "G": 143,
                        "B": 187,
                    },
                    "intensity": 1.0,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": [
                        "142Nd_CAV1",
                        "148Nd_panCK",
                        "161Dy_AQP1",
                        "164Dy_PDPN",
                        "175Lu_eCadherin",
                        "176Yb_EPCAM",
                    ],
                },
                "erythrocytes": {
                    "color": {
                        "R": 186,
                        "G": 56,
                        "B": 69,
                    },
                    "intensity": 1.0,
                    "median_filter_size": 0,
                    "gaussian_filter_sigma": 0,
                    "histogram_normalisation": False,
                    "normalize_percentage_min": 10,
                    "normalize_percentage_max": 90,
                    "targets": ["160Gd_Ter119"],
                },
            },
            "background": {
                "color": {
                    "R": 255,
                    "G": 255,
                    "B": 255,
                },
            },
        }
    return config
