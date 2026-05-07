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
home_loc = os.getenv("HOME")
if isinstance(home_loc, str):
    DATA_PATH = Path.joinpath(Path(home_loc), ".data")
    BIORED_DIR = Path.joinpath(DATA_PATH, "BioRED")
    BIOID_DIR = Path.joinpath(DATA_PATH, "BioIDtraining_2/")
    KRISSBERT_DIR = Path.joinpath(DATA_PATH, "Krissbert")
    BCD5_DIR = Path.joinpath(DATA_PATH, "BCD5")
    BIORED_CAL = Path.joinpath(BIORED_DIR, "BioRed_calibration.tsv")
    BIORED_TEST = Path.joinpath(BIORED_DIR, "BioRed_test.tsv")
    CACHED_LLM_DIR = Path.joinpath(DATA_PATH, "cached_llm_groundings")
