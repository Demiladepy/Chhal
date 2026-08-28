# Audit scripts

Reproduction scripts for the claims in the README that are easy to disbelieve. Each is
standalone and prints its own result; none writes to `results/`.

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

Run any of them with the project venv from the repo root:

    .venv/bin/python scripts/audit/real_positive_anchor.py
