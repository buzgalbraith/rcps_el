from importlib import import_module
from typing import TYPE_CHECKING

from .scorer import Scorer
from .gilda_scorer import gildaScorer
from .krissbert_scorer import krissbertScorer
from .MedCodErScorer import MedCodErScorer
from .med_path_scorer import medPathScorer
from .retrievalScorer import retrievalScorer
from .cumulativeRetrievalScorer import cumulativeRetrievalScorer

if TYPE_CHECKING:
    from .fuzzy_string_scorer import fuzzyStringScore
    from .sab_bert_scorer import sapbertScorer
    from .llm_scorer import llmScorer

## scorers with optional dependencies: name -> (module, extra) ##
_OPTIONAL_SCORERS = {
    "fuzzyStringScore": (".fuzzy_string_scorer", "fuzzy"),
    "sapbertScorer": (".sab_bert_scorer", "sapbert"),
    "llmScorer": (".llm_scorer", "llm"),
}

__all__ = [
    "Scorer",
    "gildaScorer",
    "krissbertScorer",
    "MedCodErScorer",
    "medPathScorer",
    "retrievalScorer",
    "cumulativeRetrievalScorer",
    "fuzzyStringScore",
    "sapbertScorer",
    "llmScorer",
]


def __getattr__(name):
    if name not in _OPTIONAL_SCORERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, extra = _OPTIONAL_SCORERS[name]
    try:
        module = import_module(module_name, __name__)
    except ImportError as e:
        raise ImportError(
            f"{name} requires optional dependencies, install with `pip install rcps-el[{extra}]`"
        ) from e
    return getattr(module, name)
