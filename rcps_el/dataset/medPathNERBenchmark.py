"""
MedPath with NER-predicted mention spans (rebuttal E8: how NER quality affects RCPS-EL).

Every row is a *gold* MedPath mention (kept by medPathBenchmark's hierarchy-path rule,
which looks at gold labels only), so all conditions share one risk denominator. What
changes is which span is linked:

  source "gold"  the gold span itself
  source "ner"   the NER-predicted span matched to the gold mention
                 match="overlap": the overlapping predicted span sharing the most characters
                 match="exact":   only a predicted span with identical offsets
                 no match: the mention is *missed*, its candidate set is empty, and every
                 loss scores it 1 at every threshold (a constant in lambda, so the sets stay
                 nested, the loss stays monotone, and the miss only raises R(lambda_inf))

Calibration and validation can come from different sources, which gives the three
conditions of the experiment on identical document splits:

  A  medPathNERBenchmark(calibration_source="gold", validation_source="gold")
  B  medPathNERBenchmark(calibration_source="ner",  validation_source="ner")
  C  medPathNERBenchmark(calibration_source="gold", validation_source="ner")   (mismatch)

Candidates are SapBERT top-20 lists produced by scripts/ner/run_medpath_sapbert.py
(the same retriever for gold and NER spans). NER spans come from
scripts/ner/run_medpath_ner.py. NER spans that overlap no gold mention (false positives)
have no correct label, so they are not part of the risk; they are counted in
`ner_summary`.

By default only documents the NER model was not trained on are used: CDR dev and test
(en_ner_bc5cdr_md was trained on CDR train; dev was used for model selection, but its
NER F1 matches test) and all of NCBI Disease. Documents are pooled and re-split by
document, stratified by corpus, exactly as medPathBenchmark(resplit=True) does.
"""
import json
from typing import Dict, Tuple

import polars as pl

from .dataset import Path
from .medPathBenchmark import medPathBenchmark, MAX_PREDS, logger, process_json
from rcps_el.utils.constants import MEDPATH_DIR, MEDPATH_DOCUMENT_DIR

SOURCES = ("gold", "ner")
MATCHES = ("overlap", "exact")
DEFAULT_NER_MODEL = "en_ner_bc5cdr_md"
## (corpus, splits) the NER model never saw in training ##
DEFAULT_DOCUMENTS: Dict[str, Tuple[str, ...]] = {"cdr": ("dev", "test"), "ncbi": ("train", "dev", "test")}


class medPathNERBenchmark(medPathBenchmark):
    name = "MedPathNER"
    processed_dataframe_path: Path = MEDPATH_DIR.joinpath("processed_ner")
    sapbert_dir: Path = MEDPATH_DIR.joinpath("sapbert")

    def __init__(
        self,
        calibration_source: str = "gold",
        validation_source: str = "gold",
        ner_model: str = DEFAULT_NER_MODEL,
        match: str = "overlap",
        seed: int = 100,
        split_size: float = 0.3,
        subset: list | None = None,
        k_size: int = 20,
        calibration_size: int | None = None,
        calibration_seed: int | None = None,
        documents: Dict[str, Tuple[str, ...]] | None = None,
        drop_rate: float = 0.0,
        drop_seed: int = 0,
    ) -> None:
        """
        calibration_source, validation_source : "gold" or "ner"
            Which spans are linked on each side of the split (see module docstring).
        ner_model : str
            Folder under <medpath>/ner and <medpath>/sapbert/ner_<model>.
        match : "overlap" or "exact"
            How a predicted span is matched to a gold mention.
        subset : optional list of corpora ("cdr", "ncbi")
        documents : optional {corpus: splits}
            Which shipped MedPath documents to pool. Defaults to the documents the NER
            model was not trained on.
        drop_rate : optional float in [0, 1)
            Simulated NER misses (rebuttal E8b): empty the candidate set of this fraction
            of gold mentions, on both sides of the split, so they are missed at every
            threshold. Drawn once with drop_seed over all mentions, so the same mentions
            are dropped for every split seed and a larger rate drops a superset.
        drop_seed : optional int
        seed, split_size, k_size, calibration_size, calibration_seed : as medPathBenchmark
        """
        if not 0.0 <= drop_rate < 1.0:
            raise ValueError(f"drop_rate must be in [0, 1); got {drop_rate}")
        self.drop_rate = float(drop_rate)
        self.drop_seed = int(drop_seed)
        for s in (calibration_source, validation_source):
            if s not in SOURCES:
                raise ValueError(f"source must be one of {SOURCES}; got {s!r}")
        if match not in MATCHES:
            raise ValueError(f"match must be one of {MATCHES}; got {match!r}")
        if k_size < 1 or k_size > MAX_PREDS:
            raise ValueError(f"K size must be between 1 and {MAX_PREDS}, you entered {k_size}")
        self.calibration_source = calibration_source
        self.validation_source = validation_source
        self.ner_model = ner_model
        self.match = match
        self.documents = dict(documents or DEFAULT_DOCUMENTS)
        self.subset = list(subset) if subset else list(self.documents)
        unknown = set(self.subset) - set(self.documents)
        if unknown:
            raise ValueError(f"subset {sorted(unknown)} not among documents {sorted(self.documents)}")
        ## medPathBenchmark state the inherited helpers and split_parameters rely on ##
        self.method = "medpath"
        self.resplit = True
        self.seed = seed
        self.split_size = split_size
        self.k_size = k_size
        self.calibration_size = calibration_size
        self.calibration_seed = (seed if calibration_seed is None else calibration_seed) if calibration_size else None

        ## the name carries everything that changes the rows, so the set evaluator's
        ## de-duplication never confuses two conditions ##
        self.name = f"MedPathNER_cal-{calibration_source}_val-{validation_source}"
        if "ner" in (calibration_source, validation_source):
            if match != "overlap":
                self.name += f"_{match}"
            if ner_model != DEFAULT_NER_MODEL:
                self.name += f"_{ner_model}"
        if self.documents != DEFAULT_DOCUMENTS:
            self.name += "_docs-" + "-".join(f"{c}.{'.'.join(s)}" for c, s in sorted(self.documents.items()))
        if subset:
            self.name += "_" + "_".join(self.subset)
        if self.drop_rate:
            self.name += f"_drop{self.drop_rate:g}" + (f"s{self.drop_seed}" if self.drop_seed else "")
        self.ner_summary: dict = {}
        self.preprocess_dataset()

    ## ------------------------------------------------------------------ ##
    ## building one table per source                                       ##
    ## ------------------------------------------------------------------ ##
    def _cache_key(self, source: str) -> str:
        key = "gold" if source == "gold" else f"ner_{self.ner_model}_{self.match}"
        docs = "-".join(f"{c}.{'.'.join(s)}" for c, s in sorted(self.documents.items()))
        return f"{key}__{docs}"

    def _sapbert_run(self, source: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
        run = self.sapbert_dir.joinpath("gold" if source == "gold" else f"ner_{self.ner_model}")
        mentions, candidates = run.joinpath("mentions.parquet"), run.joinpath("candidates.parquet")
        if not (mentions.exists() and candidates.exists()):
            cmd = "--source gold" if source == "gold" else f"--source ner --ner-model {self.ner_model}"
            raise FileNotFoundError(
                f"no SapBERT retrieval at {run}; run "
                f"`.venv/bin/python scripts/ner/run_medpath_sapbert.py {cmd}` first"
            )
        return pl.read_parquet(mentions), pl.read_parquet(candidates)

    def _gold_rows(self) -> pl.DataFrame:
        """gold mentions kept by medPathBenchmark's hierarchy-path rule, with offsets"""
        umls_to_mesh, _ = self._ensure_umls()
        path_curies = self._load_path_curies()
        ## document -> corpus from the files actually used (the inherited
        ## _document_corpora also opens corpora that are not on disk) ##
        corpora = {}
        for corpus, splits in self.documents.items():
            for split in splits:
                for doc in process_json(MEDPATH_DOCUMENT_DIR.joinpath(f"{corpus}_{split}.jsonl")):
                    corpora[str(doc["doc_id"])] = corpus
        records = []
        for split in sorted({s for splits in self.documents.values() for s in splits}):
            gold = self._load_gold_entities(split, umls_to_mesh, path_curies)
            for entity_id, row in gold.items():
                corpus = corpora.get(row["document_id"])
                if corpus is not None and split in self.documents[corpus]:
                    records.append({**row, "corpus": corpus, "medpath_split": split})
        rows = pl.from_dicts(records)
        ## offsets: the retrieval run lists every gold mention under the same entity ids ##
        gold_mentions, _ = self._sapbert_run("gold")
        rows = rows.join(
            gold_mentions.select(entity_id="mention_id", start="start", end="end"),
            on="entity_id",
            how="left",
            validate="1:1",
        )
        if rows["start"].null_count():
            raise ValueError(f"{rows['start'].null_count()} gold mentions have no offsets in the retrieval run")
        return rows

    def _match_ner(self, gold: pl.DataFrame) -> pl.DataFrame:
        """for each gold mention, the predicted span it is linked through (or none)"""
        ner, _ = self._sapbert_run("ner")
        docs = set(gold["document_id"])
        ner = ner.filter(pl.col("doc_id").is_in(docs))
        preds: Dict[str, list] = {}
        for r in ner.iter_rows(named=True):
            preds.setdefault(r["doc_id"], []).append((r["start"], r["end"], r["text"]))

        ## false positives are judged against *all* gold mentions, including those the
        ## hierarchy rule drops: a span on an out-of-scope mention is not spurious ##
        all_gold, _ = self._sapbert_run("gold")
        all_gold_spans: Dict[str, list] = {}
        for r in all_gold.filter(pl.col("doc_id").is_in(docs)).iter_rows(named=True):
            all_gold_spans.setdefault(r["doc_id"], []).append((r["start"], r["end"]))
        n_spurious = sum(
            not any(ps < ge and gs < pe for gs, ge in all_gold_spans.get(doc, []))
            for doc, spans in preds.items()
            for ps, pe, _ in spans
        )

        linked, status = [], []
        for r in gold.select("document_id", "start", "end").iter_rows(named=True):
            s, e = r["start"], r["end"]
            candidates = preds.get(r["document_id"], [])
            exact = [p for p in candidates if p[0] == s and p[1] == e]
            if exact:
                linked.append(exact[0][2])
                status.append("exact")
                continue
            if self.match == "overlap":
                overlapping = [p for p in candidates if p[0] < e and s < p[1]]
                if overlapping:
                    ## most shared characters, then closest length, then leftmost ##
                    best = min(
                        overlapping,
                        key=lambda p: (-(min(e, p[1]) - max(s, p[0])), abs((p[1] - p[0]) - (e - s)), p[0]),
                    )
                    linked.append(best[2])
                    status.append("overlap")
                    continue
            linked.append(None)
            status.append("missed")

        out = gold.with_columns(linked_text=pl.Series(linked, dtype=pl.String), ner_status=pl.Series(status))
        counts = out.group_by("corpus", "ner_status").len().sort("corpus", "ner_status")
        self.ner_summary = {
            "ner_model": self.ner_model,
            "match": self.match,
            "n_gold": out.height,
            "status_counts": {f"{c}/{s}": n for c, s, n in counts.iter_rows()},
            "n_predicted_spans": ner.height,
            "n_spurious_spans": int(n_spurious),
        }
        return out

    def _attach_candidates(self, rows: pl.DataFrame, source: str) -> pl.DataFrame:
        """top-MAX_PREDS SapBERT candidates for each linked span; empty for a missed mention"""
        _, umls_names = self._ensure_umls()
        path_curies = self._load_path_curies()
        _, cands = self._sapbert_run(source)
        cuis_out, names_out, scores_out = [], [], []
        for text, cuis, scores in cands.select("text", "cuis", "scores").iter_rows():
            keep_c, keep_n, keep_s = [], [], []
            ## the index is already restricted to these CUIs; re-check so the rule matches
            ## medPathBenchmark._extract_df exactly ##
            for c, sc in zip(cuis, scores):
                if c in path_curies and c in umls_names:
                    keep_c.append(f"umls:{c}")
                    keep_n.append(umls_names[c])
                    keep_s.append(float(sc))
                    if len(keep_c) >= MAX_PREDS:
                        break
            cuis_out.append(keep_c)
            names_out.append(keep_n)
            scores_out.append(keep_s)
        table = pl.DataFrame(
            {"linked_text": cands["text"], "match_curies": cuis_out, "match_names": names_out, "match_scores": scores_out},
            schema={"linked_text": pl.String, "match_curies": pl.List(pl.String),
                    "match_names": pl.List(pl.String), "match_scores": pl.List(pl.Float64)},
        )
        out = rows.join(table, on="linked_text", how="left")
        empty_s = pl.lit([], dtype=pl.List(pl.String))
        out = out.with_columns(
            pl.col("match_curies").fill_null(empty_s),
            pl.col("match_names").fill_null(empty_s),
            pl.col("match_scores").fill_null(pl.lit([], dtype=pl.List(pl.Float64))),
        )
        unlinked = out.filter(pl.col("linked_text").is_not_null() & (pl.col("match_curies").list.len() == 0)).height
        if unlinked:
            raise ValueError(f"{unlinked} linked spans have no candidates in the {source} retrieval run")
        return out

    def _source_table(self, source: str) -> pl.DataFrame:
        path = self.processed_dataframe_path.joinpath(f"{self._cache_key(source)}.parquet")
        summary_path = path.with_suffix(".json")
        if path.exists():
            if source == "ner" and summary_path.exists():
                with open(summary_path) as f:
                    self.ner_summary = json.load(f)
            return pl.read_parquet(path)
        logger.warning("building MedPath NER table for %s -> %s", source, path)
        gold = self._gold_rows()
        if source == "gold":
            rows = gold.with_columns(linked_text=pl.col("text"), ner_status=pl.lit("gold"))
        else:
            rows = self._match_ner(gold)
        table = self._attach_candidates(rows, source).select(
            "document_id", "entity_id", "text", "linked_text", "ner_status", "obj_synonyms",
            "match_names", "match_curies", "match_scores", "corpus", "medpath_split", "start", "end",
        ).sort("entity_id")
        path.parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(path)
        if source == "ner":
            with open(summary_path, "w") as f:
                json.dump(self.ner_summary, f, indent=1)
        return table

    def predicted_spans(self, document_ids) -> pl.DataFrame:
        """
        Every NER-predicted span in the given documents, with its SapBERT candidates and
        a class (for the precision analysis; the risk itself is over gold mentions):

          linked        overlaps a gold mention the risk is computed over
          out_of_scope  overlaps only gold mentions the hierarchy-path rule drops
          spurious      overlaps no gold mention (a false positive)
        """
        docs = {str(d) for d in document_ids}
        ner, _ = self._sapbert_run("ner")
        ner = ner.filter(pl.col("doc_id").is_in(docs))
        in_scope = self._source_table("gold").filter(pl.col("document_id").is_in(docs))
        all_gold, _ = self._sapbert_run("gold")
        all_gold = all_gold.filter(pl.col("doc_id").is_in(docs))

        def spans(df: pl.DataFrame, doc_col: str) -> Dict[str, list]:
            out: Dict[str, list] = {}
            for d, s, e in df.select(doc_col, "start", "end").iter_rows():
                out.setdefault(d, []).append((s, e))
            return out

        kept, every = spans(in_scope, "document_id"), spans(all_gold, "doc_id")
        overlaps = lambda ps, pe, gold: any(ps < ge and gs < pe for gs, ge in gold)  # noqa: E731
        classes = [
            "linked" if overlaps(s, e, kept.get(d, [])) else "out_of_scope" if overlaps(s, e, every.get(d, [])) else "spurious"
            for d, s, e in ner.select("doc_id", "start", "end").iter_rows()
        ]
        rows = ner.select(
            document_id="doc_id", entity_id="mention_id", text="text", linked_text="text", start="start", end="end",
        ).with_columns(span_class=pl.Series(classes, dtype=pl.String))
        return self._attach_candidates(rows, "ner")

    def _drop_mentions(self, table: pl.DataFrame) -> pl.DataFrame:
        """empty the candidate sets of a fixed random drop_rate fraction of mentions"""
        ## rank every mention by one fixed random key; the lowest drop_rate fraction is
        ## dropped, so the dropped set is nested across rates for a given drop_seed ##
        ids = table.select("entity_id").sort("entity_id").with_columns(
            key=pl.int_range(pl.len()).shuffle(seed=self.drop_seed)
        )
        n_drop = int(round(self.drop_rate * ids.height))
        dropped = ids.sort("key").head(n_drop)["entity_id"]
        is_dropped = pl.col("entity_id").is_in(dropped.implode())
        return table.with_columns(
            pl.when(is_dropped).then(pl.lit([], dtype=pl.List(pl.String))).otherwise(pl.col("match_curies")).alias("match_curies"),
            pl.when(is_dropped).then(pl.lit([], dtype=pl.List(pl.String))).otherwise(pl.col("match_names")).alias("match_names"),
            pl.when(is_dropped).then(pl.lit([], dtype=pl.List(pl.Float64))).otherwise(pl.col("match_scores")).alias("match_scores"),
            pl.when(is_dropped).then(pl.lit("dropped")).otherwise(pl.col("ner_status")).alias("ner_status"),
        )

    ## ------------------------------------------------------------------ ##
    ## splits                                                              ##
    ## ------------------------------------------------------------------ ##
    def preprocess_dataset(self) -> None:
        tables = {s: self._source_table(s) for s in {self.calibration_source, self.validation_source}}
        tables = {s: t.filter(pl.col("corpus").is_in(self.subset)) for s, t in tables.items()}
        if len(tables) == 2 and set(tables["gold"]["entity_id"]) != set(tables["ner"]["entity_id"]):
            raise ValueError("gold and NER tables hold different gold mentions")
        if self.drop_rate:
            tables = {s: self._drop_mentions(t) for s, t in tables.items()}
        ## document-level split, stratified by corpus, as medPathBenchmark._stratified_resplit ##
        documents = next(iter(tables.values())).select("document_id", "corpus").unique().sort("document_id")
        validation_ids = []
        for _, docs in documents.group_by("corpus", maintain_order=True):
            shuffled = docs.sort("document_id").sample(fraction=1.0, shuffle=True, seed=self.seed)
            validation_ids += shuffled.head(int(self.split_size * shuffled.height))["document_id"].to_list()
        is_validation = pl.col("document_id").is_in(validation_ids)
        self.calibration_set = tables[self.calibration_source].filter(~is_validation).with_row_index()
        self.validation_set = tables[self.validation_source].filter(is_validation).with_row_index()
        self.test_set = None
        self.calibration_set = self._top_k_cols(self.calibration_set)
        self.validation_set = self._top_k_cols(self.validation_set)
        if self.calibration_size:
            self.calibration_set = self._calibration_subsample()
