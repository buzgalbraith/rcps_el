"""
Loader for BioRed benchmark
"""

from .dataset import Dataset, pl, Path
from rcps_og.utils import safeMatch
from bioregistry import normalize_curie
import gilda
from indra.databases import hgnc_client, mesh_client
from rcps_og.utils.constants import BIORED_DIR
import os
import logging

logger = logging.getLogger(__name__)


def get_equivalent_curies(curie):
    """Get all equivalent CURIEs for an entity"""
    # Could use PyOBO, BioThings, or INDRA's built-in mappings
    prefix, identifier = curie.split(":")

    equivalents = set([curie])

    # Example: for HGNC, get NCBI Gene xrefs
    if prefix == "hgnc":
        gene_id = hgnc_client.get_entrez_id(identifier)
        # print(f'gene_id {gene_id}')
        if gene_id:
            equivalents.add(f"ncbigene:{gene_id}")
    elif prefix == "ncbigene":
        gene_id = hgnc_client.get_hgnc_from_entrez(identifier)
        # print(f'gene_id {gene_id}')
        if gene_id:
            equivalents.add(f"hgnc:{gene_id}")
    elif prefix == "mesh":
        # mesh_id = mesh_client.get_mesh_id_from_db_id(db_ns='chebi', db_id='37684')
        # print(curie)
        res = mesh_client.get_db_mapping(identifier)
        if res:
            a, b = res
            res_2 = normalize_curie(f"{a}:{b}")
            equivalents.add(res_2)
    # else:
    #     res = mesh_client.get_mesh_id_from_db_id(db_ns='ncbigene', db_id='50489')
    return list(equivalents)


class bioRedBenchmark(Dataset):
    name = "bioRED"
    document_id_column = "document_id"
    original_dataframe_path: Path = BIORED_DIR.joinpath("BioRed_calibration.tsv")
    processed_dataframe_path: Path = BIORED_DIR.joinpath(
        "processed_BioRed_calibration.parquet"
    )

    def __init__(self, seed: int = 100, split_size: float = 0.2) -> None:
        super().__init__(seed, split_size)

    def load_dataframe(self, dataframe_path: Path | None = None) -> pl.DataFrame:
        if dataframe_path is None:
            dataframe_path = self.original_dataframe_path
        return pl.read_csv(dataframe_path, separator="\t")

    def get_gilda_candidates(self, text: str, context: str | None) -> list[safeMatch]:
        matches = gilda.ground(text=text, context=context)
        records = []
        for match in matches:
            curie = normalize_curie(f"{match.term.db}:{match.term.id}")
            records.append(
                {
                    "name": match.term.entry_name,
                    "curie": curie,
                    "score": match.score,
                }
            )
            equivalences = get_equivalent_curies(curie)
            if equivalences:
                for eqc in equivalences:
                    records.append(
                        {
                            "name": match.term.entry_name,
                            "curie": eqc,
                            "score": match.score,
                        }
                    )
        return records

    def preprocess_dataset(
        self,
    ) -> pl.DataFrame:
        """pre-process the dataset"""
        if os.path.exists(self.processed_dataframe_path):
            logger.info(
                f"Loading dataset from cache at {self.processed_dataframe_path}"
            )
            return pl.read_parquet(self.processed_dataframe_path)
        df = (
            self.original_dataframe.with_columns(
                gilda_matches=pl.struct(["entity_raw_text", "full_text"]).map_elements(
                    lambda x: self.get_gilda_candidates(
                        x["entity_raw_text"], x["full_text"]
                    ),
                    return_dtype=pl.List(
                        pl.Struct(
                            {
                                "name": pl.String,
                                "curie": pl.String,
                                "score": pl.Float64,
                            }
                        )
                    ),
                )
            )
            .with_columns(
                match_names=pl.col("gilda_matches").list.eval(
                    pl.element().struct.field("name")
                ),
                match_curies=pl.col("gilda_matches").list.eval(
                    pl.element().struct.field("curie")
                ),
                gilda_scores=pl.col("gilda_matches").list.eval(
                    pl.element().struct.field("score")
                ),
                obj_synonyms=pl.struct(["db", "identifier"]).map_elements(
                    lambda x: [normalize_curie(":".join([x["db"], x["identifier"]]))],
                    return_dtype=pl.List(pl.String),
                ),
            )
            .rename({"entity_raw_text": "text"})
        )
        df = df.with_columns(pl.col("obj_synonyms").explode()).with_columns(
            pl.col("obj_synonyms")
            .map_elements(get_equivalent_curies, return_dtype=pl.List(pl.String))
            .alias("obj_synonyms")
        )

        df.write_parquet(self.processed_dataframe_path)
        return df
