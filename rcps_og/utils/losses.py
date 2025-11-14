"""
Loss functions. Likely want to make these instances of a class.
"""


def binary_miscoverage_loss(
    entity: dict, candidate: dict, size_weighted: bool = False
) -> float:
    """
    Classic confromal binary miscoverage loss.
    Args:
        entity(dict): dict with the string name for each candidate
        candidate(dict): dict with the true label
        size_weighted(optional, bool): If true loss will be weighted by size of candidate set
    Returns:
        loss (float): Loss value
    """
    label = entity.get("normalized_name", None)
    candidate_set = [x.get("entry_name", "") for x in candidate]
    bml = float(label not in candidate_set)
    return len(candidate_set) * bml if size_weighted else bml
