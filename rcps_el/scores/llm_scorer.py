"""
LLM Scorer class
"""

from .scorer import Scorer, pl
from dglink.core.LLMClients import ollamaClient, openAIClient, LLMClient
from rcps_el.utils.constants import CACHED_LLM_DIR
import textwrap
import json
from pydantic import BaseModel
import tqdm
from gilda.process import normalize, replace_greek_spelled_out
import os

## data frame schemas ##
DF_SCHEMA = pl.Schema(
    {
        "term": pl.String,
        "grounding": pl.String,
        "probability": pl.Float64,
    }
)
DF_SCHEMA_TITLE = pl.Schema(
    {
        "term": pl.String,
        "title": pl.String,
        "grounding": pl.String,
        "probability": pl.Float64,
    }
)


class llmScorer(Scorer):
    raw_text_col = "text"
    entity_name_col = "match_names"
    title_col = "title"

    def __init__(self, use_titles: bool = False, batch_size: int = 15):
        super().__init__()
        self.llm_client = ollamaClient()
        self.use_titles = use_titles
        self.batch_size = batch_size
        title_str = "_with_title" if use_titles else ""
        os.makedirs(CACHED_LLM_DIR, exist_ok=True)
        safe_model_name = self.llm_client.model.replace(":", "_").replace("-", "_")
        self.cache_path = CACHED_LLM_DIR.joinpath(
            f"{safe_model_name}_cached_annotations_batch_size_{self.batch_size}{title_str}.tsv",
        )
        if self.use_titles:
            self.key_cols = ["term", "title", "grounding"]
        else:
            self.key_cols = ["term", "grounding"]
        self.name = f"LLM_scorer{title_str}_batch_size{self.batch_size}"
        self._load_cached_groundings()

    ## here needed to overload execute to work with potential tittles ##
    def execute(self, data_frame: pl.DataFrame) -> pl.DataFrame:
        """Get scores over a dataset"""
        self.processing_function(data_frame=data_frame)
        if not self.use_titles:
            return data_frame.with_columns(
                pl.struct([self.raw_text_col, self.entity_name_col])
                .map_elements(
                    lambda x: self.score_sample(
                        x[self.raw_text_col], x[self.entity_name_col]
                    ),
                    return_dtype=pl.List(pl.Float64),
                )
                .alias(f"{self.name}")
            )
        else:
            return data_frame.with_columns(
                pl.struct(
                    [
                        self.raw_text_col,
                        self.entity_name_col,
                        self.title_col,
                    ]
                )
                .map_elements(
                    lambda x: self.score_sample(
                        x[self.raw_text_col],
                        x[self.entity_name_col],
                        x[self.title_col],
                    ),
                    return_dtype=pl.List(pl.Float64),
                )
                .alias(f"{self.name}")
            )

    def score_sample(
        self,
        entity: str,
        candidates: list[str],
        title: str | None = None,
    ) -> list[float]:
        """Score candidates against entity, optionally using titles."""

        responses = []
        for candidate in candidates:
            normed = (self.full_norm(entity),)
            if self.use_titles:
                normed += (self.full_norm(title),)
            normed += (self.full_norm(candidate),)

            if normed in self.cached_groundings:
                responses.append(self.cached_groundings[normed])
            else:
                print(f"running LLM for {normed}")
                new_obs_df = self._call_model([normed])
                self._update_cache(new_obs_df)
                responses.append(new_obs_df["probability"][0])

        return responses

    def processing_function(self, data_frame: pl.DataFrame, max_retries=15):
        def check_key(df: pl.DataFrame, key: tuple):
            """checks if a key is present in the dataframe"""
            if len(key) == 3:
                sec = df.filter(
                    pl.col("term").eq(x[0])
                    & pl.col("title").eq(x[1])
                    & pl.col("grounding").eq(x[2])
                )
            else:
                sec = df.filter(pl.col("term").eq(x[0]) & pl.col("grounding").eq(x[1]))
            return len(sec) == 0

        cached_groundings_df = self._load_cached_df()
        run_list = self._construct_prompt_set(input_df=data_frame)
        missed_list = self._check_missed_terms(
            run_list=run_list, evaluation_df=cached_groundings_df
        )
        batch_size = self.batch_size
        print(f"Starting with {len(missed_list)} terms and batch size {batch_size}")
        try_count = 0
        while len(missed_list) > 0 and try_count < max_retries:
            batch_size = max(1, batch_size - 1)
            for i in tqdm.tqdm(range(0, len(missed_list), batch_size)):
                to_run = []
                for x in missed_list[i : i + batch_size]:
                    if check_key(cached_groundings_df, key=x):
                        to_run.append(x)
                if len(to_run) > 0:
                    results_df = self._call_model(to_run)
                    cached_groundings_df = (
                        cached_groundings_df.vstack(results_df)
                        .group_by(self.key_cols)
                        .first()
                    )
                    if (i // batch_size) % 5 == 0:
                        cached_groundings_df.write_csv(
                            self.cache_path,
                            separator="\t",
                        )
            missed_list = self._check_missed_terms(
                run_list=run_list, evaluation_df=cached_groundings_df
            )
            try_count += 1
            print(
                f"missed {len(missed_list)} terms at batch size {batch_size} for try {try_count}"
            )
            cached_groundings_df.write_csv(
                self.cache_path,
                separator="\t",
            )

    def full_norm(self, x):
        """full gilda style name normalization"""
        return normalize(replace_greek_spelled_out(x))

    def _update_cache(self, new_obs_df: pl.DataFrame) -> None:
        """Persist a new observation to the cache and reload."""
        (
            self._load_cached_df()
            .vstack(new_obs_df)
            .group_by(self.key_cols)
            .first()
            .write_csv(self.cache_path, separator="\t")
        )
        self._load_cached_groundings()

    def _load_cached_df(self) -> pl.DataFrame:
        """load cached data frame from disk"""
        if self.use_titles:
            schema = DF_SCHEMA_TITLE
        else:
            schema = DF_SCHEMA
        if os.path.exists(self.cache_path):
            return pl.read_csv(self.cache_path, separator="\t", schema=schema)
        return pl.DataFrame(schema=schema)

    def _load_cached_groundings(self) -> dict[tuple[str, str], float]:
        """load cached groundings as a dict"""
        self.cached_groundings = dict()
        cached_df = self._load_cached_df()
        for row in cached_df.iter_rows(named=True):
            self.cached_groundings[tuple(row[key] for key in self.key_cols)] = row[
                "probability"
            ]

    def _construct_prompt_set(self, input_df: pl.DataFrame) -> list:
        """get a set of unique prompts given to LLM"""
        run_set = set()
        for row in input_df.iter_rows(named=True):
            term = row[self.raw_text_col]
            title = row[self.title_col]
            raw_groundings = row[self.entity_name_col]
            for grounding in raw_groundings:
                if self.use_titles:
                    run_set.add(
                        (
                            self.full_norm(term),
                            self.full_norm(title),
                            self.full_norm(grounding),
                        )
                    )
                else:
                    run_set.add((self.full_norm(term), self.full_norm(grounding)))
        return sorted(list(run_set))

    def _call_model(self, batch: list[tuple], single_call=False):
        if not self.use_titles:
            system_prompt = textwrap.dedent(f"""
                    You are a scientist checking biomedical entity linking.
                    For each term-grounding pair below, return your 
                    probability estimate that the grounding is correct
                    as well as the term and grounding
                """).strip()
            df_schema = DF_SCHEMA
            pydantic_schema = GroundingProbabilityBatched
        else:
            system_prompt = textwrap.dedent(f"""
                    You are a scientist checking biomedical entity linking.
                    For each term, abstract title, grounding triple below, return your 
                    probability estimate that the grounding is correct
                    as well as the term, title and grounding
                """).strip()
            df_schema = DF_SCHEMA_TITLE
            pydantic_schema = GroundingProbabilityWithTitleBatched

        user_prompt = textwrap.dedent(f"""
                term grounding pair: 
                {json.dumps(batch, indent=2)}
            """).strip()
        llm_resp = self.llm_client.structured_call(
            context=system_prompt,
            user_prompt=user_prompt,
            schema=pydantic_schema,
            max_retries=3,
        )
        ## short circuit if calling model for one input ##
        if single_call:
            return llm_resp.grounding_probabilities[0].probability
        call_result = []
        for item in llm_resp.grounding_probabilities:
            item_dict = item.model_dump()
            call_result.append(item_dict)
            normed = tuple(self.full_norm(item_dict.get(key)) for key in self.key_cols)
            if not (normed) in batch:
                print(f"Warning {(normed)} is weird...")
        return pl.from_dicts(call_result, schema=df_schema)

    def _check_missed_terms(
        self, run_list: list, evaluation_df: pl.DataFrame
    ) -> list[tuple]:
        """Run through and check missed terms from a current run."""
        if self.use_titles:
            cols = ["term", "title", "grounding"]
        else:
            cols = ["term", "grounding"]

        prompt_df = pl.DataFrame(run_list, schema=cols, orient="row")
        missed = prompt_df.join(evaluation_df, on=cols, how="anti")
        return missed.rows()


## pydantic classes ##
class GroundingProbability(BaseModel):
    term: str
    grounding: str
    probability: float


class GroundingProbabilityWithTitle(BaseModel):
    term: str
    title: str
    grounding: str
    probability: float


class GroundingProbabilityBatched(BaseModel):
    grounding_probabilities: list[GroundingProbability]


class GroundingProbabilityWithTitleBatched(BaseModel):
    grounding_probabilities: list[GroundingProbabilityWithTitle]
