"""
MedPath benchmark 
"""
from .dataset import Dataset, pl, Path
from rcps_el.utils.constants import MEDPATH_DOCUMENT_DIR, MEDPATH_PATH_FILES


import pystow
import datasets
from bioregistry import normalize_curie
from indra.databases.mesh_client import get_mesh_tree_numbers

import os
import json
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)
module = pystow.module("MedPath")


SPLIT_MAP = {
    'train' : 'calibration',
    'dev' : 'validation',
    'test' : 'test'
}

PROCESSED_SCHEMA = {
    "document_id": pl.String,
    "entity_id": pl.String,
    "text": pl.String,
    "obj_synonyms": pl.List(pl.String),
    "match_names": pl.List(pl.String),
    "match_curies": pl.List(pl.String),
    "match_scores": pl.List(pl.Float64),
}


CORPORA = ("cdr", "ncbi", "cometa", 'medmentions') 
## max candidate set size we are considering ## 
MAX_PREDS = 20
MIN_CANDIDATES = MAX_PREDS

## bigbio data frame format ##

KB_FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "document_id": datasets.Value("string"),
        "passages": [
            {
                "id": datasets.Value("string"),
                "type": datasets.Value("string"),
                "text": datasets.Sequence(datasets.Value("string")),
                "offsets": datasets.Sequence([datasets.Value("int32")]),
            }
        ],
        "entities": [
            {
                "id": datasets.Value("string"),
                "type": datasets.Value("string"),
                "text": datasets.Sequence(datasets.Value("string")),
                "offsets": datasets.Sequence([datasets.Value("int32")]),
                "normalized": [
                    {
                        "db_name": datasets.Value("string"),
                        "db_id": datasets.Value("string"),
                    }
                ],
            }
        ],
        "events": [
            {
                "id": datasets.Value("string"),
                "type": datasets.Value("string"),
                "trigger": {
                    "text": datasets.Sequence(datasets.Value("string")),
                    "offsets": datasets.Sequence([datasets.Value("int32")]),
                },
                "arguments": [
                    {
                        "role": datasets.Value("string"),
                        "ref_id": datasets.Value("string"),
                    }
                ],
            }
        ],
        "coreferences": [
            {
                "id": datasets.Value("string"),
                "entity_ids": datasets.Sequence(datasets.Value("string")),
            }
        ],
        "relations": [
            {
                "id": datasets.Value("string"),
                "type": datasets.Value("string"),
                "arg1_id": datasets.Value("string"),
                "arg2_id": datasets.Value("string"),
                "normalized": [
                    {
                        "db_name": datasets.Value("string"),
                        "db_id": datasets.Value("string"),
                    }
                ],
            }
        ],
    }
)

## helper functions for parsing raw training data to big bio format ## 

def _native_ontology_to_db_name(native_ontology_name: str) -> str:
    mapping = {"MSH": "MESH", "MESH": "MESH", "OMIM": "OMIM"}
    return mapping.get(native_ontology_name.upper(), native_ontology_name)
def convert_document(doc: Dict, doc_index: int) -> Dict:
    doc_id = str(doc.get("doc_id", doc_index))
    text = doc.get("text", "")

    passages = [
        {
            "id": f"{doc_id}-text",
            "type": "text",
            "text": [text],
            "offsets": [[0, len(text)]],
        }
    ]

    entities = []
    for i, m in enumerate(doc.get("mentions", [])):
        normalized = []
        if m.get("native_id"):
            normalized.append(
                {
                    "db_name": _native_ontology_to_db_name(
                        m.get("native_ontology_name", "")
                    ),
                    "db_id": m["native_id"],
                }
            )
        if m.get("cui"):
            normalized.append({"db_name": "UMLS", "db_id": m["cui"]})

        entities.append(
            {
                "id": f"{doc_id}-entity-{i}",
                "type": m.get("entity_type", ""),
                "text": [m["text"]],
                "offsets": [[m["start"], m["end"]]],
                "normalized": normalized,
            }
        )

    return {
        "id": doc_id,
        "document_id": doc_id,
        "passages": passages,
        "entities": entities,
        "events": [],
        "coreferences": [],
        "relations": [],
    }


def build_dataset(documents: List[Dict]) -> datasets.Dataset:
    records = [convert_document(d, i) for i, d in enumerate(documents)]
    return datasets.Dataset.from_list(records, features=KB_FEATURES)


def process_json(raw_path):
    jsons = []
    with open(raw_path, mode='r') as f:
        for line in f:
            line = line.strip()
            if line:
                jsons.append(json.loads(line))
    return jsons

def pct(numerator: int, denominator: int) -> float:
    return (numerator / denominator) * 100 if denominator else float("nan")


class medPathBenchmark(Dataset):
    """MedPath mentions grounded to UMLS, with SapBERT candidate lists.

    Only entities we can build a hierarchy path for are kept, and only entities with
    a full candidate list survive -- the RCPS evaluator trims candidates by score, so
    every row has to start from the same candidate budget.
    """

    name = "MedPath"
    document_id_column = "document_id"
    original_dataframe_path: Path = module.base.joinpath("raw_predictions", "json")
    processed_dataframe_path: Path = module.base.joinpath("processed_predictions")
    known_methods = ["medpath"]

    def __init__(
        self,
        seed: int = 100,
        split_size: float = 0.2,
        method: str = "medpath",
        original_dataframe_path: str = None,
        resplit: bool = False,
        subset: list = None, 
    ) -> None:
        """
        resplit : optional, bool
            By default calibrate on the MedPath train split and validate on dev. Those
            are not exchangeable (the train split carries far more NCBI-disease
            documents than dev), which the RCPS guarantee requires. With resplit,
            pool train/dev/test and draw a document-level split_size fraction of each
            corpus for validation (seeded), so both sets share one distribution.
        subset : optional list
            By default uses all available corpus, but can also use just a subset
        """
        self.seed = seed
        self.split_size = split_size
        self.resplit = resplit
        if resplit:
            logger.info(f"Respiting dataset for better class ballance ")
            self.name += "_resplit"
        self.subset = subset or CORPORA
        assert all(src.lower() in CORPORA for src in self.subset), f"Can not use subset:{self.subset} known corpus are {CORPORA}"

        if subset:
            logger.info(f"filter for only mentions from {self.subset}")
            self.name += '_'.join([""] + list(self.subset))
        self.method = method.lower().strip()
        assert (
            self.method in self.known_methods
        ), f"Method: {self.method} not available known methods for dataset {self.name} are {self.known_methods}"
        if original_dataframe_path:
            self.original_dataframe_path = Path(original_dataframe_path)
        self.preprocess_dataset()
    
    ## build umls artifacts ## 
    def _build_umls_caches(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """build and store UMLS names and mappings
        """
        from openacme.icd10.map_definitions import _ensure_umls_files

        mrconso_path, _ = _ensure_umls_files()
        umls_to_mesh: Dict[str, str] = {}
        umls_id_to_name: Dict[str, str] = {}
        preferred = set()
        logger.warning("Building UMLS caches from %s (one pass, this is slow)", mrconso_path)
        with open(Path(mrconso_path).as_posix(), mode="r") as f:
            for line in f:
                raw = line.rstrip("\n").split("|")
                umls_id, lang, ts, stt, ispref, sab, code = (
                    raw[0], raw[1], raw[2], raw[4], raw[6], raw[11], raw[13]
                )
                if sab == "MESH" and umls_id not in umls_to_mesh:
                    umls_to_mesh[umls_id] = code
                if lang != "ENG":
                    continue
                name = raw[14] 
                ## by default use preferred name ## 
                if (ts == "P") and (stt == "PF") and (ispref == "Y"):
                    umls_id_to_name[umls_id] = name
                    preferred.add(umls_id)
                ## if can not find prefered ty another ## 
                elif umls_id not in preferred and umls_id not in umls_id_to_name:
                    umls_id_to_name[umls_id] = name
        return umls_to_mesh, umls_id_to_name

    def _ensure_umls(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """build or load UMLS mappings"""
        if getattr(self, "_umls_cache", None) is not None:
            return self._umls_cache
        xwalk_path = module.base.joinpath("umls", "umls_mesh_xwalk.parquet")
        name_path = module.base.joinpath("umls", "umls_names.parquet")
        if xwalk_path.exists() and name_path.exists():
            logger.info("Loading UMLS caches from %s", xwalk_path.parent)
            xwalk = pl.read_parquet(xwalk_path)
            names = pl.read_parquet(name_path)
            self._umls_cache = (
                dict(zip(xwalk["CUI"], xwalk["CODE"])),
                dict(zip(names["CUI"], names["STR"])),
            )
            return self._umls_cache
        umls_to_mesh, umls_id_to_name = self._build_umls_caches()
        xwalk_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {"CUI": list(umls_to_mesh.keys()), "CODE": list(umls_to_mesh.values())}
        ).write_parquet(xwalk_path)
        pl.DataFrame(
            {"CUI": list(umls_id_to_name.keys()), "STR": list(umls_id_to_name.values())}
        ).write_parquet(name_path)
        self._umls_cache = (umls_to_mesh, umls_id_to_name)
        return self._umls_cache

    def _load_path_curies(self) -> set:
        """every curie MedPath has already built a hierarchy path for"""
        if getattr(self, "_path_curies", None) is not None:
            return self._path_curies
        curies = set()
        for path in MEDPATH_PATH_FILES:
            with open(path, mode="r") as f:
                curies.update(json.loads(f.read().strip()))
        self._path_curies = curies
        return curies

    def _check_for_path(
        self, umls_id: str, umls_to_mesh: Dict[str, str], medpath_curies: set
    ) -> bool:
        """Check if MedPath has a path, or if a path exists in the mesh tree structure"""
        if umls_id in medpath_curies:
            return True
        mesh_id = umls_to_mesh.get(umls_id, None)
        if not mesh_id:
            return False
        return len(get_mesh_tree_numbers(mesh_id=mesh_id)) > 0

    def _safe_curie(self, db_name: str, db_id: str) -> str | None:
        """normalize_curie to bioregistry"""
        curie = normalize_curie(f"{db_name}:{db_id}")
        if curie is None:
            logger.warning("could not normalize curie %s:%s, dropping", db_name, db_id)
        return curie


    def raw_path(self, split: str) -> Path:
        return self.original_dataframe_path.joinpath(f"{split}_predictions.json")

    def processed_path(self, split: str) -> Path:
        return self.processed_dataframe_path.joinpath(f"medpath_{SPLIT_MAP[split]}.parquet")

    def _load_gold_entities(
        self, split: str, umls_to_mesh: Dict[str, str], medpath_curies: set
    ) -> Dict[str, Dict]:
        """gold mentions for a split, keyed by the entity ids the predictions use"""
        docs = []
        for path in sorted(MEDPATH_DOCUMENT_DIR.iterdir()):
            if path.stem.rsplit("_", 1)[-1] == split and path.stem.startswith(CORPORA):
                logger.info("loading %s", path)
                docs += process_json(path)
        if not docs:
            raise FileNotFoundError(
                f"no {CORPORA} documents for split={split} under {MEDPATH_DOCUMENT_DIR}"
            )

        gold = {}
        keepers = 0
        dropers = 0
        for doc in build_dataset(docs):
            document_id = doc.get("id")
            for entity in doc.get("entities") or []:
                entity_id = entity.get("id")
                keep = False
                obj_synonyms = set()
                for norm in entity.get("normalized") or []:
                    db_name = norm.get("db_name")
                    db_id = norm.get("db_id")
                    curie = self._safe_curie(db_name, db_id)
                    if curie is not None:
                        obj_synonyms.add(curie)
                    if (db_name == "MESH") and (len(get_mesh_tree_numbers(db_id)) > 0):
                        keep = True
                    elif (db_name == "UMLS") and self._check_for_path(
                        db_id, umls_to_mesh, medpath_curies
                    ):
                        keep = True
                if not keep:
                    dropers += 1
                    continue
                if entity_id in gold:
                    raise ValueError(
                        f"duplicate entity id {entity_id} -- document ids collide across {CORPORA}"
                    )
                keepers += 1
                gold[entity_id] = {
                    "document_id": document_id,
                    "entity_id": entity_id,
                    "text": entity.get("text")[0],
                    "obj_synonyms": sorted(obj_synonyms),
                }
        logger.warning(
            "%s: entities with a hierarchy path %d/%d (%.2f%%)",
            split,
            keepers,
            keepers + dropers,
            pct(keepers, keepers + dropers),
        )
        return gold

    def _extract_df(self, split: str) -> pl.DataFrame:
        """build one split: gold mentions joined to their SapBERT candidate lists.

        The raw prediction files are several hundred MB with ~400 candidates per
        entity, so they are streamed a document at a time and each candidate list is
        truncated as it is filtered -- nothing holds the full file in memory.
        """
        umls_to_mesh, umls_id_to_name = self._ensure_umls()
        medpath_curies = self._load_path_curies()
        gold = self._load_gold_entities(split, umls_to_mesh, medpath_curies)

        records = []
        seen = set()
        keepers_norm = 0
        path_drops = 0
        name_drops = 0
        len_drops = 0
        with open(self.raw_path(split), mode="r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                prediction = json.loads(line)
                for entity in prediction.get("entities") or []:
                    entity_id = entity.get("id")
                    ## skip entities with no gold row, and any repeat of one we did ##
                    if entity_id not in gold or entity_id in seen:
                        continue
                    seen.add(entity_id)
                    match_curies = []
                    match_names = []
                    match_scores = []
                    for norm in entity.get("normalized") or []:
                        if len(match_curies) >= MAX_PREDS:
                            break
                        umls_id = norm.get("db_id")
                        if umls_id not in medpath_curies:
                            path_drops += 1
                            continue
                        umls_name = umls_id_to_name.get(umls_id, None)
                        if umls_name is None:
                            name_drops += 1
                            continue
                        curie = self._safe_curie("umls", umls_id)
                        if curie is None:
                            continue
                        keepers_norm += 1
                        match_curies.append(curie)
                        match_names.append(umls_name)
                        match_scores.append(norm.get("score"))
                    ## every row starts from the same candidate budget, so the
                    ## evaluator's set-size numbers stay comparable across rows ##
                    if len(match_curies) < MIN_CANDIDATES:
                        len_drops += 1
                        continue
                    record = dict(gold[entity_id])
                    record["match_names"] = match_names
                    record["match_curies"] = match_curies
                    record["match_scores"] = match_scores
                    records.append(record)

        unmatched = len(gold) - len(seen)
        if unmatched:
            logger.warning("%s: %d gold entities had no prediction row, dropped", split, unmatched)
        logger.warning(
            "%s: candidates kept %.2f%%, entities with a full candidate list %.2f%%",
            split,
            pct(keepers_norm, keepers_norm + name_drops + path_drops),
            pct(len(records), len(records) + len_drops),
        )
        if not records:
            raise ValueError(f"no {self.name} rows survived preprocessing for split={split}")
        df = pl.from_records(records, schema=PROCESSED_SCHEMA).with_row_index()
        corpora = self._document_corpora()
        return df.with_columns(corpus=pl.col("document_id").replace_strict(corpora))
    def preprocess_dataset(self) -> None:
        """pre-process the dataset

        MedPath ships its own train/dev/test document splits and we have predictions
        for all three, so use them directly rather than re-splitting one of them --
        that keeps documents from leaking between calibration and validation.
        """
        for split in SPLIT_MAP:
            output_path = self.processed_path(split)
            if not os.path.exists(output_path):
                logger.warning(f"Loading {split}")
                df = self._extract_df(split)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.write_parquet(output_path)
                logger.warning(f"{self.raw_path(split)} extracted to {output_path}")
        self.calibration_set = pl.read_parquet(self.processed_path("train"))
        self.validation_set = pl.read_parquet(self.processed_path("dev"))
        self.test_set = pl.read_parquet(self.processed_path("test"))
        ## if desired take only certain corpus ##
        self._subset_dataframe()
        if self.resplit:
            self.calibration_set, self.validation_set = self._stratified_resplit()
            ## every document is now in calibration or validation ##
            self.test_set = None
    def _subset_dataframe(self) -> None:
        """Filter dataset to only selected subset"""
        self.calibration_set = self.calibration_set.filter(pl.col('corpus').is_in(self.subset))
        self.validation_set = self.validation_set.filter(pl.col('corpus').is_in(self.subset))
        self.test_set = self.test_set.filter(pl.col('corpus').is_in(self.subset))
    def _document_corpora(self) -> Dict[str, str]:
        """document id -> source corpus, read from the raw MedPath documents"""
        corpora = {}
        for corpus in CORPORA:
            for split in SPLIT_MAP:
                for doc in process_json(MEDPATH_DOCUMENT_DIR.joinpath(f"{corpus}_{split}.jsonl")):
                    doc_id = str(doc.get("doc_id"))
                    if corpora.setdefault(doc_id, corpus) != corpus:
                        raise ValueError(f"document {doc_id} appears in more than one corpus")
        return corpora

    def _stratified_resplit(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """pool the shipped splits and re-split documents, stratified by corpus"""
        pooled = pl.concat([self.calibration_set, self.validation_set, self.test_set]).drop("index")
        if pooled["entity_id"].n_unique() != pooled.height:
            raise ValueError("entity ids collide across the shipped MedPath splits")
        corpora = self._document_corpora()
        documents = (
            pooled.select("document_id").unique().sort("document_id")
            .with_columns(corpus=pl.col("document_id").replace_strict(corpora))
        )
        validation_ids = []
        for corpus, docs in documents.group_by("corpus", maintain_order=True):
            shuffled = docs.sort("document_id").sample(fraction=1.0, shuffle=True, seed=self.seed)
            validation_ids += shuffled.head(int(self.split_size * shuffled.height))["document_id"].to_list()
        is_validation = pl.col("document_id").is_in(validation_ids)
        return (
            pooled.filter(~is_validation).with_row_index(),
            pooled.filter(is_validation).with_row_index(),
        )

    def load_dataframe(self, dataframe_path: Path | None = None) -> pl.DataFrame:
        if not dataframe_path:
            return self.calibration_set
        return pl.read_parquet(dataframe_path)
