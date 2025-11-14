from pathlib import Path
import os

ENTITY_TYPE_MAPS = {
    "CellLine": "cellosaurus",
    "ChemicalEntity": "mesh",
    "DiseaseOrPhenotypicFeature": "mesh",  # note this one might use omim as well
    "GeneOrGeneProduct": "ncbigene",
    "OrganismTaxon": "ncbitaxon",
    "SequenceVariant": "dbSNP",
}

RESOURCE_PATH = Path.joinpath(Path(os.getenv("HOME")), ".data/BioRED")
CALIBRATION_DATA_PATH = Path.joinpath(RESOURCE_PATH, "BioRed_calibration.tsv")
TESTING_DATA_PATH = Path.joinpath(RESOURCE_PATH, "BioRed_test.tsv")
