# Audit scripts

Reproduction scripts for the claims in the README that are easy to disbelieve. Each is
standalone and prints its own result. Most write nothing; the two that produce a table
worth keeping write a single CSV to `results/` and are marked below.

    real_positive_anchor.py          Real-fraud recall @0.1% FPR next to each vector's
                                     recall BEFORE and AFTER the evasion optimizer.
                                     This is the paper's central table.

    narrowness_matched_ks.py         mean_ks_controlled for narrow REAL fraud subgroups,
                                     against the vectors. Shows the headline comparison
                                     is only fair when narrowness is matched.

    ks_null_floor.py                 Per-feature legit-vs-legit KS. The floor is not zero
                                     and not constant, so a hard-coded threshold means
                                     different things at different n.

    degradation_ratio.py             KS normalised by that floor, per population.

    subgroup_separability_control.py Null control: is a narrow real subgroup trivially
                                     separable from the rest of real fraud anyway?

    generator_fingerprint.py         AUC of synthetic attacks vs REAL fraud. If this is
                                     high, the loop is learning the generator, not the
                                     threat.

    vector_separability.py           Pairwise AUC between every pair of vectors on the
                                     ten controlled columns. Refutes the idea that
                                     upi_collect is a sibling of bustout (0.997), and
                                     shows the sharper thing: every pair separates at
                                     >= 0.957 while the detector catches all six at
                                     0.00%. Separability and detectability are not the
                                     same measurement.
                                     -> results/vector_separability.csv

    card_precision_at_k.py           CP@k per day, averaged over days. The operational
                                     metric, computed once so it exists, and deliberately
                                     not the headline: it flatters this system in exactly
                                     the direction the repo argues against.
                                     -> results/card_precision_at_k.csv

Run any of them with the project venv from the repo root:

    .venv/bin/python scripts/audit/real_positive_anchor.py
