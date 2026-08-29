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

    acp_vocabulary.py                Re-derives every claim in the agentic-commerce
                                     section from the live ACP spec: six specs not one,
                                     a closed single-member RiskSignal enum unchanged
                                     across every published version, no pacing vocabulary
                                     anywhere, and Channel.type = [browser] with
                                     BrowserInfo required. Needs network; touches nothing
                                     else. Exists because the original claim was written
                                     from one file out of six.

    why_the_attacks_score_zero.py    Why all six vectors score 0.00%. Kills the obvious
                                     off-support explanation (swapping all ten CONTROLLED
                                     columns for real fraud's values leaves recall at
                                     0.00%) and finds the real one: swap the sixteen
                                     INHERITED columns instead and detectability comes
                                     straight back. The host-selection rule is the whole
                                     left-hand end of the arms-race curve.
                                     -> results/inherited_block_transplant.csv,
                                        results/card_testing_offsupport.csv

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

    trajectory_replay_probe.py       Does copying a victim's real trajectory beat
                                     resampling its marginals? Prediction written down
                                     first and held: both sit at the floor as generated.
                                     Finds that only 7.5% of campaigns have a victim with
                                     enough history to replay at all, so it also runs on
                                     a pool gated to hosts that can. There the copy is
                                     the more faithful sequence (gap CV 0.70 vs 0.38
                                     against an ideal 1.00) and buys no evasion: paired
                                     difference under the experiment-E transplant is
                                     -0.07% +- 0.12%. A null, and a well-powered one.
                                     -> results/trajectory_replay_detection.csv,
                                        results/trajectory_replay_sequence.csv

Run any of them with the project venv from the repo root:

    .venv/bin/python scripts/audit/real_positive_anchor.py
