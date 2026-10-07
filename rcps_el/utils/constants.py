from pathlib import Path
from pystow import module

ENTITY_TYPE_MAPS = {
    "CellLine": "cellosaurus",
    "ChemicalEntity": "mesh",
    "DiseaseOrPhenotypicFeature": "mesh",  # note this one might use omim as well
    "GeneOrGeneProduct": "ncbigene",
    "OrganismTaxon": "ncbitaxon",
    "SequenceVariant": "dbSNP",
}


BIORED_DIR = module("rcps_el", "BioRED").base
BIOID_DIR = module("rcps_el", "BioIDtraining_2").base
KRISSBERT_DIR = module("rcps_el", "Krissbert").base
MEDCODER_DIR = module("rcps_el", "medcoder").base
BCD5_DIR = module("rcps_el", "BCD5").base
CACHED_LLM_DIR = module("rcps_el", "cached_llm_groundings").base
MEDPATH_DIR = module("rcps_el", "medpath").base
BIORED_CAL = Path.joinpath(BIORED_DIR, "BioRed_calibration.tsv")
BIORED_TEST = Path.joinpath(BIORED_DIR, "BioRed_test.tsv")
MEDPATH_DOCUMENT_DIR = Path.joinpath(MEDPATH_DIR, "documents")
MEDPATH_PATH_DIR = Path.joinpath(MEDPATH_DIR, "hierarchical_paths")
MEDPATH_PATH_FILES = [
    Path.joinpath(MEDPATH_PATH_DIR,  "GO_paths.json"),
    Path.joinpath(MEDPATH_PATH_DIR,  "HPO_paths.json"),
    Path.joinpath(MEDPATH_PATH_DIR, "MSH_paths.json"),
    Path.joinpath(MEDPATH_PATH_DIR, "LNC_paths.json"),
    Path.joinpath(MEDPATH_PATH_DIR, "NCBI_paths.json"),
]

