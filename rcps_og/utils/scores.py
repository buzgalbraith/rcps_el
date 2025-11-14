"""
Scoring functions. Likely want to make these instances of a class.
"""

from rapidfuzz import fuzz


def gilda_score(entity: dict, candidate: dict) -> float:
    """simple score that just takes gilda's estimate"""
    return candidate.get("gilda_score", None)


def fuzzy_string_score(entity: dict, candidate: dict) -> float:
    """
    fuzzy string matching score.
    """
    x = entity.get("entity_raw_text", None)
    y = candidate.get("entry_name", None)
    if (x is None) or (y is None):
        return 0
    else:
        return fuzz.ratio(x, y) / 100
