"""
Zeshel zero-shot entity linking benchmark (Logeswaran et al., 2019), a general
domain (Fandom wiki) counterpart to the biomedical benchmarks.

Candidates are precomputed by scripts/zeshel_bm25.py, one parquet per
(split, world). Mentions are split into calibration/validation at random by
their context document, so the two halves are exchangeable.
"""

from .dataset import Dataset, pl, Path
from rcps_el.utils.constants import ZESHEL_DIR


class zeshelBenchmark(Dataset):
    name = "Zeshel"
    document_id_column = "context_document_id"
    known_methods = ["bm25", "biencoder"]

    def __init__(
        self,
        worlds: list[str] = ["muppets"],
        zeshel_split: str = "val",
        seed: int = 100,
        split_size: float = 0.5,
        method: str = "bm25",
    ) -> None:
        self.worlds = worlds
        self.zeshel_split = zeshel_split
        if len(worlds) == 1:
            self.name = f"Zeshel ({worlds[0]})"
        super().__init__(seed=seed, split_size=split_size, method=method)

    def load_dataframe(self, dataframe_path: Path | None = None) -> pl.DataFrame:
        paths = [
            ZESHEL_DIR / f"zeshel_{self.method}_{self.zeshel_split}_{w}.parquet"
            for w in self.worlds
        ]
        missing = [str(p) for p in paths if not p.exists()]
        assert not missing, f"Missing candidates {missing}; run scripts/zeshel_bm25.py first"
        return pl.concat([pl.read_parquet(p) for p in paths])

    def preprocess_dataset(self) -> pl.DataFrame:
        ## context document ids are unique per world, but namespace them to be safe ##
        return self.original_dataframe.with_columns(
            pl.concat_str(["corpus", "context_document_id"], separator=":").alias(
                "context_document_id"
            )
        )
