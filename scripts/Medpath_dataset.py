

import datasets
import polars as pl
from indra.databases.mesh_client import get_mesh_tree_numbers
from bioregistry import normalize_curie
from pystow import module

import json
import logging
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


SPLIT_MAP = {
    'train' : 'calibration',
    'dev' : 'validation',
    'test' : 'test'
}

MEDPATH_ROOT = Path("/Users/buzgalbraith/workspace/MedPath")
DOCUMENT_DIR = MEDPATH_ROOT.joinpath("data_processed", "documents")
PATH_FILES = [
    MEDPATH_ROOT.joinpath("data_processed/hierarchical_paths/go/results/GO_paths.json"),
    MEDPATH_ROOT.joinpath("data_processed/hierarchical_paths/hpo/results/HPO_paths.json"),
    MEDPATH_ROOT.joinpath("data_processed/hierarchical_paths/mesh/results/MSH_paths.json"),
    MEDPATH_ROOT.joinpath("data_processed/hierarchical_paths/loinc/results/LNC_paths.json"),
    MEDPATH_ROOT.joinpath("data_processed/hierarchical_paths/ncbi/results/NCBI_paths.json"),
]
UMLS_XWALK_PATH = Path("umls.txt")

## only splits with a predictions file can be built -- the prediction ids are what
## the candidate lists are keyed on. add an entry here to process another split. ##
PREDICTION_PATHS = {
    'dev' : Path("json/dev_predictions.json"),
}

## corpora covered by the predictions files above ##
CORPORA = ("cdr", "ncbi")

OUTPUT_MODULE = module("medpath")

## stop collecting candidates once we have this many, and drop the entity if we
## could not reach it -- one threshold, two roles, so name both. ##
MAX_PREDS = 20
MIN_CANDIDATES = MAX_PREDS

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


def safe_curie(db_name: str, db_id: str) -> str | None:
    """normalize_curie returns None for prefixes bioregistry does not know --
    keep those out of the output instead of writing nulls into the curie lists."""
    curie = normalize_curie(f"{db_name}:{db_id}")
    if curie is None:
        logger.warning("could not normalize curie %s:%s, dropping", db_name, db_id)
    return curie


def pct(numerator: int, denominator: int) -> float:
    return (numerator / denominator) * 100 if denominator else float("nan")


def matches_split(path: Path, split: str) -> bool:
    """Exact match on the token after the last underscore in the file
    stem, instead of endswith -- endswith('test.jsonl') also matches a
    file literally named 'latest.jsonl' (confirmed: 'latest.jsonl'
    .endswith('test.jsonl') is True). Adjust the split logic here if your
    filenames don't follow a '..._<split>.jsonl' convention."""
    return path.stem.rsplit("_", 1)[-1] == split


def _prepare_paths():
    """
    load umls to mesh xwalk and paths from medpath
    """
    umls_xwalk = pl.read_csv(UMLS_XWALK_PATH)
    umls_to_mesh = {}
    for row in umls_xwalk.iter_rows(named=True):
        umls_to_mesh[row.get("CUI")] = row.get("CODE")
    curies = set()
    for pth in PATH_FILES:
        with open(pth, mode = 'r') as f:
            raw = f.read().strip()
            jsn = json.loads(raw)
            for x in jsn:
                curies.add(x)
    return umls_to_mesh, curies


def check_for_path(umls_id:str, umls_to_mesh:Dict, medpath_curies:set)->bool:
    """checks if we have a path for a given umls term from (1) MedPaths paths (2) Indra MeSH Client (3) UMLS Saps"""
    if umls_id in medpath_curies:
        return True
    mesh_id = umls_to_mesh.get(umls_id, None)
    if not mesh_id:
        return False
    return len(get_mesh_tree_numbers(mesh_id=mesh_id)) > 0


def load_umls_names() -> Dict[str, str]:
    """CUI -> preferred english name, out of MRCONSO.RRF.

    Streams the file (MRCONSO is multi-GB) and prefers the row flagged as the
    canonical atom (TS=P, STT=PF, ISPREF=Y), falling back to the first english
    row for a CUI so that subset files without a preferred atom still resolve.
    """
    umls_name_path = module("openacme").base.joinpath("umls", "MRCONSO.RRF")
    umls_id_to_name = {}
    preferred = set()
    with open(umls_name_path.as_posix(), mode='r') as f:
        for line in f:
            raw = line.rstrip("\n").split("|")
            umls_id, lang, ts, stt, ispref = raw[0], raw[1], raw[2], raw[4], raw[6]
            if lang != "ENG":
                continue
            name = raw[14]  ## STR, by column index -- raw[-5] only worked via the trailing '|' ##
            is_preferred = (ts == "P") and (stt == "PF") and (ispref == "Y")
            if is_preferred:
                umls_id_to_name[umls_id] = name
                preferred.add(umls_id)
            elif umls_id not in preferred and umls_id not in umls_id_to_name:
                umls_id_to_name[umls_id] = name
    return umls_id_to_name


def build_split(split: str) -> datasets.Dataset | None:
    """build the KB dataset for a single split -- returns None if nothing matched"""
    docs = []
    for x in DOCUMENT_DIR.iterdir():
        if matches_split(x, split) and str(x.stem).startswith(CORPORA):
            print(f"loading {x}")
            docs += process_json(str(x))
    if not docs:
        print(f"[warn] no files matched split={split}, skipping")
        return None
    return build_dataset(docs)


def collect_candidates(ds, umls_to_mesh: Dict, medpath_curies: set) -> Dict[str, Dict]:
    """keep the entities we have some hierarchy path for"""
    candidates = {}
    keepers = 0
    dropers = 0
    for doc in ds:
        document_id = doc.get("id")
        for entity in doc.get("entities") or []:
            text = entity.get("text")[0]
            entity_id = entity.get("id")
            keep = False
            obj_synonyms = set()
            for norm in entity.get("normalized") or []:
                db_name = norm.get("db_name")
                db_id = norm.get("db_id")
                curie = safe_curie(db_name, db_id)
                if curie is not None:
                    obj_synonyms.add(curie)
                if (db_name == "MESH") and (len(get_mesh_tree_numbers(db_id)) > 0):
                    keep = True
                elif (db_name == "UMLS") and check_for_path(db_id, umls_to_mesh, medpath_curies):
                    keep = True
            if not keep:
                dropers += 1
                continue
            ## entity ids are '<doc_id>-entity-<i>', so two corpora sharing a doc_id
            ## would silently overwrite each other. fail instead of corrupting. ##
            if entity_id in candidates:
                raise ValueError(
                    f"duplicate entity id {entity_id} -- document ids collide across corpora {CORPORA}"
                )
            keepers += 1
            candidates[entity_id] = {
                'document_id' : document_id,
                'entity_id' : entity_id,
                'text' : text,
                'obj_synonyms' : list(obj_synonyms),
            }
    print(f"entity keep rate: {pct(keepers, keepers + dropers):.2f}")
    return candidates


def attach_predictions(
    candidates: Dict[str, Dict], predictions_path: Path, umls_id_to_name: Dict, medpath_curies: set
) -> List[Dict]:
    """join candidate lists onto the kept entities. only entities that end up with a
    full candidate list are returned -- an entity with no prediction row at all is
    dropped the same way a short one is, rather than going out with null match fields."""
    keepers_norm = 0
    keepers_entity = 0
    path_drops = 0
    name_drops = 0
    len_drops = 0
    records = []
    seen = set()
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prediction = json.loads(line)
            for entity in prediction.get("entities") or []:
                ## skip entities we do not have ##
                entity_id = entity.get("id")
                if entity_id not in candidates or entity_id in seen:
                    continue
                has_ground_truth = False
                seen.add(entity_id)
                match_curies = []
                match_names = []
                match_scores = []
                for norm in entity.get("normalized") or []:
                    if len(match_curies) >= MAX_PREDS:
                        break
                    umls_id = norm.get("db_id")
                    if not umls_id in medpath_curies:
                        path_drops += 1
                        continue
                    umls_name = umls_id_to_name.get(umls_id, None)
                    if umls_name is None:
                        name_drops += 1
                        continue
                    curie = safe_curie("umls", umls_id)
                    if curie is None:
                        continue
                    keepers_norm += 1
                    match_curies.append(curie)
                    match_names.append(umls_name)
                    match_scores.append(norm.get('score'))

                if len(match_curies) < MIN_CANDIDATES:
                    len_drops += 1
                    continue
                keepers_entity += 1
                record = dict(candidates[entity_id])
                record['match_curies'] = match_curies
                record['match_names'] = match_names
                record['match_scores'] = match_scores
                records.append(record)

    ## entities with no row in the predictions file at all -- previously these stayed
    ## in the output with null match columns ##
    unmatched = len(candidates) - len(seen)
    if unmatched:
        print(f"[warn] {unmatched} kept entities had no prediction row, dropped")
    print(f"normalization keep rate: {pct(keepers_norm, keepers_norm + name_drops + path_drops):.2f}")
    print(f"entity candidate keep rate: {pct(keepers_entity, keepers_entity + len_drops):.2f}")
    return records


if __name__ == "__main__":
    umls_to_mesh, medpath_curies = _prepare_paths()
    umls_id_to_name = load_umls_names()

    for split, predictions_path in PREDICTION_PATHS.items():
        ## load rows and labels dataset ##
        ds = build_split(split)
        if ds is None:
            continue

        ## check if we have some path for the terms ##
        candidates = collect_candidates(ds, umls_to_mesh, medpath_curies)

        ## get relevant predictions ##
        records = attach_predictions(candidates, predictions_path, umls_id_to_name, medpath_curies)
        if not records:
            print(f"[warn] no records survived for split={split}, nothing written")
            continue

        ## convert to list and write out ##
        records_df = pl.from_records(records).with_row_index()
        output_path = OUTPUT_MODULE.base.joinpath(f"medpath_{SPLIT_MAP[split]}.parquet")
        records_df.write_parquet(output_path)
        print(f"wrote {records_df.height} rows to {output_path}")