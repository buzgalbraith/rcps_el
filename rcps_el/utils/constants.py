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
    MEDPATH_DIR = Path.joinpath(DATA_PATH, "MedPath")
    MEDPATH_DOCUMENT_DIR = Path.joinpath(MEDPATH_DIR, "documents")
    MEDPATH_PATH_DIR = Path.joinpath(MEDPATH_DIR, "hierarchical_paths")
    MEDPATH_PATH_FILES = [
        Path.joinpath(MEDPATH_PATH_DIR,  "GO_paths.json"),
        Path.joinpath(MEDPATH_PATH_DIR,  "HPO_paths.json"),
        Path.joinpath(MEDPATH_PATH_DIR, "MSH_paths.json"),
        Path.joinpath(MEDPATH_PATH_DIR, "LNC_paths.json"),
        Path.joinpath(MEDPATH_PATH_DIR, "NCBI_paths.json"),
    ]

    ## MedPath is a checkout rather than a download, so allow an override ##
    # MEDPATH_DIR = Path(
    #     os.getenv("MEDPATH_DIR", Path.joinpath(Path(home_loc), "workspace", "MedPath"))
    # )
    # MEDPATH_DOCUMENT_DIR = Path.joinpath(MEDPATH_DIR, "data_processed", "documents")
    # MEDPATH_PATH_DIR = Path.joinpath(MEDPATH_DIR, "data_processed", "hierarchical_paths")
    # MEDPATH_PATH_FILES = [
    #     Path.joinpath(MEDPATH_PATH_DIR, "go", "results", "GO_paths.json"),
    #     Path.joinpath(MEDPATH_PATH_DIR, "hpo", "results", "HPO_paths.json"),
    #     Path.joinpath(MEDPATH_PATH_DIR, "mesh", "results", "MSH_paths.json"),
    #     Path.joinpath(MEDPATH_PATH_DIR, "loinc", "results", "LNC_paths.json"),
    #     Path.joinpath(MEDPATH_PATH_DIR, "ncbi", "results", "NCBI_paths.json"),
    # ]
