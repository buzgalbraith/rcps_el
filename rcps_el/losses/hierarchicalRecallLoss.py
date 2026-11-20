"""
Hierarchical recall loss (Kiritchenko, Matwin, Nock & Famili, 2006).

Each label is augmented with all of its ancestors (root excluded):
    C     = Anc+(y)                      true label and its ancestors
    C_hat = union_{y_hat in T} Anc+(y_hat)  candidates and their ancestors
    hR = |C n C_hat| / |C|,   hP = |C n C_hat| / |C_hat|

1 - hR lies in [0, 1] and is nonincreasing as candidates are added (C_hat only
grows), so it is a valid RCPS loss. It is zero iff the set covers every
ancestor of y (y itself or a descendant), and gives partial credit to
siblings / cousins proportional to the depth they share with y.

1 - hP (and 1 - hF) are NOT monotone in the set -- adding an unrelated
candidate grows C_hat without growing C n C_hat -- so they are only reported
as quality metrics (see `hierarchical_pr`), never calibrated against.
"""

from .hierarchicalLossFunction import hierarchicalLossFunction, Aggregator, safeMinAggregator

import math


class hierarchicalRecallLoss(hierarchicalLossFunction):
    """1 - hierarchical recall of the candidate set, per source ontology"""

    label_curie_col = "obj_synonyms"
    candidate_curie_col = "match_curies"
    default_agg_method: Aggregator = safeMinAggregator()

    def __init__(self, agg_method: Aggregator = None, exclude_root: bool = True):
        super().__init__(agg_method)
        self.exclude_root = exclude_root
        self._ancestor_cache: dict[tuple[str, str], frozenset] = {}
        self._native_anc: dict[str, dict[str, set]] = {}
        self._root_codes: dict[str, set] = {}
        self.name = "hR loss"

    ## ancestor sets ##
    def _native_ancestors(self, src: str) -> dict[str, set]:
        """native code -> itself and every ancestor seen on any stored path in src.

        MedPath stores only some root paths per concept, so a child can be listed
        under one parent of a multi-parent node. Pooling every path prefix that
        contains a code gives the transitive closure the stored data supports.
        """
        if src in self._native_anc:
            return self._native_anc[src]
        anc: dict[str, set] = {}
        for record in self.path_index.values():
            if src not in record:
                continue
            for path in record[src]["paths"]:
                for i, code in enumerate(path):
                    anc.setdefault(code, set()).update(path[: i + 1])
        self._native_anc[src] = anc
        return anc

    def _ancestor_set(self, umls_id: str, src: str) -> frozenset:
        """the concept and all its ancestors in src (DAG closure), root(s) excluded"""
        key = (umls_id, src)
        cached = self._ancestor_cache.get(key)
        if cached is not None:
            return cached
        native_anc = self._native_ancestors(src)
        codes, roots = set(), set()
        for path in self.path_index[umls_id][src]["paths"]:
            roots.add(path[0])
            for code in path:
                codes |= native_anc.get(code, {code})
        if self.exclude_root:
            codes -= roots
            codes -= self._roots(src)
        result = frozenset(codes)
        self._ancestor_cache[key] = result
        return result

    def _roots(self, src: str) -> set:
        """first code of every stored path in src"""
        if src not in self._root_codes:
            self._root_codes[src] = {
                path[0]
                for record in self.path_index.values() if src in record
                for path in record[src]["paths"]
            }
        return self._root_codes[src]

    def _target(self, labels: list[str]) -> str:
        umls_labels = [x.removeprefix("umls:") for x in labels if x.startswith("umls:")]
        if len(umls_labels) == 0:
            raise ValueError(f"No UMLS code present {labels}")
        if len(umls_labels) > 1:
            raise ValueError(f"Multiple UMLS codes for {labels}")
        return umls_labels[0]

    def _candidates_in(self, src: str, umls_candidates: list[str]) -> list[str]:
        """candidates that have a path in src"""
        return [
            x for x in umls_candidates
            if x in self.path_index and src in self.path_index[x]
        ]

    def _src_counts(self, target: str, src: str, umls_candidates: list[str]):
        """(|C n C_hat|, |C|, |C_hat|, target_hit) for one source ontology"""
        true_set = self._ancestor_set(target, src)
        cands = self._candidates_in(src, umls_candidates)
        pred_set = set()
        for c in cands:
            pred_set |= self._ancestor_set(c, src)
        native_target = self.path_index[target][src]["native_id"]
        target_hit = any(
            self.path_index[c][src]["native_id"] == native_target for c in cands
        )
        return len(true_set & pred_set), len(true_set), len(pred_set), target_hit

    ## loss ##
    def calc_loss(self, labels: list[str], candidate_set: list[str]) -> float:
        target = self._target(labels)
        umls_candidates = [x.removeprefix("umls:") for x in candidate_set]
        src_losses = []
        for src in self.path_index[target]:
            inter, n_true, _, target_hit = self._src_counts(target, src, umls_candidates)
            if n_true == 0:
                ## target is a root: only an exact hit counts ##
                src_losses.append(float(not target_hit))
            else:
                src_losses.append(1.0 - inter / n_true)
        return self.agg_method.execute(src_losses)

    ## reporting ##
    def hierarchical_pr(self, labels: list[str], candidate_set: list[str]) -> tuple[float, float, float]:
        """(hP, hR, hF) for one mention, taking the best source ontology by hF.

        hP is NaN for an empty set (no predictions), so it can be excluded from means.
        """
        target = self._target(labels)
        umls_candidates = [x.removeprefix("umls:") for x in candidate_set]
        best = (math.nan, 0.0, 0.0)
        for src in self.path_index[target]:
            inter, n_true, n_pred, target_hit = self._src_counts(target, src, umls_candidates)
            if n_true == 0:
                h_r = float(target_hit)
            else:
                h_r = inter / n_true
            h_p = inter / n_pred if n_pred else math.nan
            if n_true == 0 and n_pred:
                h_p = float(target_hit)
            h_f = (
                2 * h_p * h_r / (h_p + h_r)
                if not math.isnan(h_p) and (h_p + h_r) > 0
                else 0.0
            )
            if math.isnan(best[0]) or h_f > best[2]:
                best = (h_p, h_r, h_f)
        return best
