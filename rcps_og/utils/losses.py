"""
Loss functions. Likely want to make these instances of a class.
"""


def binary_miscoverage_loss(entity: dict, candidate: dict) -> float:
    """
    Classic confromal binary miscoverage loss.
    Args:
        entity(dict): dict with the string name for each candidate
        candidate(dict): dict with the true label
    Returns:
        loss (float): Loss value
    """
    label = (
        entity.get("normalized_name", "").lower()
        if entity.get("normalized_name", "") is not None
        else ""
    )
    candidate_set = [
        x.get("entry_name", "").lower()
        for x in candidate
        if x.get("entry_name", "") is not None
    ]
    return float(label not in candidate_set)


## this fails because it is not monotonic in set size.
def weighted_binary_miscoverage_loss(
    entity: dict,
    candidate: dict,
) -> float:
    """
    Classic confromal binary miscoverage loss weighted by number of candidates
    Args:
        entity(dict): dict with the string name for each candidate
        candidate(dict): dict with the true label
    Returns:
        loss (float): Loss value
    """
    label = entity.get("normalized_name", None)
    candidate_set = [x.get("entry_name", "") for x in candidate]
    bml = float(label not in candidate_set)
    return len(candidate_set) * bml
