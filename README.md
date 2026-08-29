# Chhal — a closed-loop adversarial engine for GenAI payment fraud

**Mastercard Innovation Challenge @ GFF 2026 · AI Defense Lab for Payment Security**

**Landing page:** [`web/index.html`](web/index.html) — a self-contained pitch page (serve it,
or host it on GitHub Pages).

> *Chhal* (छल) — *deception*. Every deception the attacker invents becomes the training
> ground for a defence that learns to see through it, and every gap the defence reveals
> feeds the next deception.

Most submissions are three disconnected things: a slide of attack ideas, a data
generator, and a fraud classifier. The brief says what actually wins — *"the best
solutions turn their own simulated attacks into the training ground for a stronger
defense."* So Chhal is **one closed loop**, and the loop itself is the demo.

```
        ┌─────────────────────────────────────────────────────────┐
        │                                                         │
   [1] RED-TEAM AGENT ──► [2] CONSTRAINED EVASION ──► [3] DETECTOR
   (6 GenAI attack           OPTIMIZER                  (LightGBM,
    vectors, tabular)     (evade + stay plausible)      flags fraud)
        ▲                                                         │
        │                                                         ▼
        └────────  held-out novel attacks feed back  ◄──── retrain + score
```

---

## How to read this repo

There are a lot of honesty notes below. Twelve caveats of equal loudness reads as not
knowing which result is the result, so they are ranked here once and the ranking holds
throughout.

**The finding, and the only thing worth taking away:** *a closed red/blue loop measures its
own generator at both ends.* Attack recall climbs 0.00% → 80.80% over eight rounds while
recall on real fraud sits flat at ~14% and ends where it started. The 0% is not evasion and
the 80% is not detection. Without a real-fraud control line in the same file, that curve is
unfalsifiable — and this is a general property of synthetic red teams, not a defect of this
one. [Read it here.](#what-the-arms-race-curve-actually-measures)

**Three supporting results, in order of how much they cost us:**

1. **A quarter of the detector's apparent skill was memorised entities.** Purging accounts
   that straddle the temporal split takes real-fraud recall 19.10% → 14.22%, a 25.5%
   relative drop. [→](#leakage-rules-all-enforced-in-code)
2. **Distance from legitimate traffic does not predict detection.** Real fraud and our
   vectors occupy the same distance band; one is caught 0.2–23.1% of the time and the other
   0.00%. Provenance separates them, not stealth. [→](#fidelity-of-simulation--measured-not-claimed)
3. **The detector reads *who is being paid*, not *how the sequence behaves*.** Found twice
   independently — in the `mule_fanout` ablation and in the dunning control, where flipping
   one binary column moves the false-positive rate on legitimate subscription retries from
   1.5% to 38.8%. [→](#the-confusable-class--what-card_testings-97-actually-costs)

**Three things that are limitations rather than findings**, and should be read as the
boundary of the work, not as results: linkage counts are frozen at the host's last observed
value; the coordination signal `mule_fanout` was built to test is not representable in the
frozen feature space at all; and the mitigation economics rests on invented constants and is
a second paper (only [the segment table](#whose-fraud-is-being-avoided) survives).

**Everything else in this file is method, provenance, or a negative result kept because it
was expensive to learn.** Nothing below is claimed as a contribution unless it is in the
list above or in [What is not ours](#what-is-not-ours), which names the borrowings.

---

## Where this sits in the literature

Written before the results rather than after, because most of what is uncomfortable below
has already been said by someone else, and not saying so is how these findings become a
reviewer's findings.

**On evaluations that flatter themselves.** Arp et al., *Dos and Don'ts of Machine Learning
in Computer Security* (USENIX Security 2022), catalogue the pitfalls this repo kept walking
into — sampling bias, spurious correlations, inappropriate baselines, and *lab-only
evaluation*, which is precisely what a synthetic red team produces. Pendlebury et al.,
**TESSERACT** (USENIX Security 2019), showed that removing temporal and spatial bias from a
malware evaluation collapses reported performance; the 7-day delay period and the entity
purge here are the payments analogue, and they cost 25.5% of measured skill.

**On adaptive attacks.** Tramèr et al., *On Adaptive Attacks to Adversarial Example
Defenses* (NeurIPS 2020), broke thirteen published defences by adapting the attack to each
one, and their conclusion is the discipline this loop needs: a defence evaluated only
against the attack its authors imagined is not evaluated. Dyrmishi et al. (IEEE S&P 2023)
carry that into constrained domains and find that hardening against *unrealistic*
adversarial examples transfers poorly to realistic ones — which is the risk our
`EvasionOptimizer`'s manifold guardrail exists to manage, and which the control line above
suggests it does not fully solve.

**On the attack itself.** Carminati et al., ACM TOPS 21(3) 2018, ran per-victim mimicry
against a real bank's detector with amount and timestamp as the attacker's variables —
`threshold_hugging` is that attack, and we claim the measurement, not the idea. The RAID
2020 follow-up states the immutable-feature rule this repo's `DERIVED_FEATURES` boundary
implements.

**On generators.** Sajja (arXiv:2604.13125, Apr 2026) supplies the noise-floor anchoring
used throughout the fidelity section, and proves that row-independent generators cannot
preserve within-entity inter-event-time autocorrelation — a result that indicts our own
`mimic_host` sampler and is the reason `trajectory_replay` is the one attack idea here with
a defensible claim to novelty.

**What is left.** Every one of those papers is about a *classifier* being evaluated
dishonestly. The gap this repo lands in is one step earlier: what happens when the
adversarial data itself is the thing that has not been validated, and the arms-race curve
built on it is read as progress. That is the contribution, and it is a measurement rather
than a method.

---

## The money chart: three lines, and the third one is the finding

The single result that proves novelty, efficacy, and the closed loop at once — and the
one a judge will challenge as *"circular."* It isn't. We plot **two** things and keep them
separate:

- **Blue generalisation (the money line).** A **fixed** benchmark of hard adaptive attacks
  is built once against the baseline detector and **never trained on**. A static detector
  catches almost none of it (they're optimised to look normal). After one loop pass the
  retrained detector holds it near the top — **generalisation to unseen adaptive fraud, not
  memorisation.**
- **Red pressure (the dotted line).** Each iteration the red team optimises a *fresh* batch
  against the *current* detector and holds it out. This line stays volatile — the red team
  keeps finding new evasions the just-retrained model partly misses. The **gap** between the
  lines is the honest, unfinished arms race.
- **Real IEEE-CIS fraud (the grey control line).** The same detector, the same threshold,
  scored against the real frauds in the held-out test split — a population the loop never
  touches. This line is the reason the other two can be believed or disbelieved, and it is
  the one that carries the finding below.

**No leakage:** the base split is temporal, carries a **7-day delay period** between train
and test, and **purges every account that appears on both sides** before any attack is
injected; the benchmark and pressure attacks touch neither split; the train/held-out split
is by campaign, not by row; and every run writes a **computed** leakage audit into
`results/summary.json` rather than asking you to believe this sentence.

### Measured where a payments team would actually run it

Not at `score >= 0.5`, and not in ROC AUC. Fraud systems are tuned to a **false-positive
budget** — flag no more than X% of good customers, catch as much as possible inside that —
because flagging good customers is the expensive failure. And at 3.5% prevalence ROC AUC is
flattered by an enormous true-negative pile: 0.9999 there is unremarkable. So the headline
is **recall at a fixed FPR** and **PR AUC** (average precision), with the 0.5-threshold
numbers kept only for comparison.

Full run, 8 iterations on real IEEE-CIS (`scripts/run_loop.py`, 80s), on the split with the
delay period and the entity purge applied — 442,905 train / 76,617 test:

| metric | baseline | after the loop | budget realised |
|---|---|---|---|
| recall @ **0.1%** of legit flagged | 0.00% | **80.80%** | 0.0998% |
| recall @ 0.5% | 0.13% | 91.07% | 0.4989% |
| recall @ 1.0% | 0.13% | 93.10% | 0.9992% |
| **PR AUC** | 0.0313 | **0.9508** | — |
| alert rate, real traffic only | — | 0.774% | — |
| alert rate, scored mixture | — | 3.279% | — |

The last column is there because a threshold can quote a budget it does not honour. A
gradient-boosted ensemble emits large blocks of identical scores, and a quantile can land
inside one; flagging with `>=` then sweeps the whole block in. Before this was fixed, one
configuration reported recall "at a 0.1% budget" while really flagging **43.9%** of
legitimate traffic. `threshold_for_fpr` now takes the lowest score that actually fits, and
every run writes the realised rate into `results/summary.json` so it can be checked rather
than trusted.

The two alert rates are also separate on purpose. The 3.279% divides by the scored mixture
— real legitimate traffic *plus however many attacks this run generated* — so it moves with
a config knob: the same detector reported 0.16%, 0.80% and 3.19% as `benchmark_per_vector`
went 100 → 500 → 2000. The 0.774% is measured on the real test set alone and is the one an
ops team would staff against.

For comparison, the naive 0.5 cutoff on the same run: F1 0.0023 → 0.8808, ROC AUC 0.9936, FP on
legit 0.71%. The ROC number is the one to distrust.

The baseline catching almost nothing is not a broken detector — the benchmark attacks
were optimised specifically to evade it, which is the optimizer doing its job.

The claim that survives scrutiny is not this curve. It is the leave-one-out one, and it
is much weaker: **43.1% on an attack family never seen in any form**, against 66.1% on
families the detector was trained on — a **23.0-point** generalisation gap
(`scripts/generalisation_check.py`, six vectors, post-purge split). Per family, held out:

| held out | recall if trained on it | recall never seen |
|---|---|---|
| `mule_fanout` | 60.4% | 69.6% |
| `upi_collect` | 61.0% | 64.6% |
| `bustout` | 64.2% | 51.4% |
| `threshold_hugging` | 73.2% | 36.0% |
| `autopay_mandate` | 77.7% | 24.6% |
| `card_testing` | 59.9% | **12.2%** |

A detector trained on five fraud families does not thereby understand the sixth. But the
ordering is not the one an earlier version of this file predicted — it claimed the
stealthiest vector generalises worst, and that is now false. **`card_testing`, the loudest
and most separable vector in the suite (KS 0.513, 44.6× the noise floor), is the one the
detector fails hardest to reach when it has never seen it.** Distance from legitimate
traffic does not predict transfer here any better than it predicts detection.

Two vectors — `mule_fanout` and `upi_collect` — score *higher* unseen than seen. That is
not a good sign either: it means the other five carry enough of their signature that
training on the vector itself adds nothing, which is what a shared generator fingerprint
looks like from the inside.

An earlier version of this file reported 88.7% here. That number was an artifact: the
evasion optimizer used to perturb four derived timeline features as independent scalars,
which stamped every vector with the same impossible-velocity signature, so a "never seen"
family was not really unfamiliar — it shared the tell. Fixing the optimizer (see
[Attacks are campaigns, not rows](#attacks-are-campaigns-not-rows)) cost 13 points of
headline recall and 49 points of leave-one-out recall. Both were ours to lose.

Numbers vary by seed. Every ± in this README is the standard error of a paired per-seed
difference over five seeds, not the spread of a single run — see
[the ablation](#two-of-these-do-something-the-other-three-do-not--and-one-ablation-that-did-not-go-our-way)
for what happened the one time we quoted a single-seed number.

### What the arms-race curve actually measures

The table above is the least trustworthy thing in this README, and the grey control line is
why. Run the same eight iterations and watch both numbers:

| iteration | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| **synthetic attack recall** | 0.00% | 54.53% | 65.20% | 71.07% | 73.23% | 79.67% | 81.10% | 80.60% | **80.80%** |
| **real IEEE-CIS fraud recall** | 14.22% | 16.44% | 15.89% | 15.78% | 14.22% | 15.60% | 14.68% | 13.52% | **15.02%** |

Attack recall climbs eighty points. Real-fraud recall wanders inside a 2.9-point band and
ends where it started. **Eight rounds of adversarial retraining bought zero improvement in
detecting actual fraud.**

Both ends of the curve measure the generator rather than the threat, and each end fails in
its own way:

- **The `0%` at iteration 0 is not evasion.** All six vectors score 0.00% *before* the
  optimizer runs and 0.00% *after* it runs against the un-retrained detector
  (`scripts/audit/real_positive_anchor.py`). The optimizer has nothing to improve on. Those
  rows sit in a region of feature space where the trees have no training data at all, so
  they receive a low score by default rather than by design.
- **The `80.43%` is fingerprint recognition.** Once the detector is retrained on this
  generator's output it learns the generator, and the benchmark — however carefully held
  out — was produced by the same generator.

The control rules out the charitable reading. If the loop were teaching the detector
something general about fraud-shaped behaviour, the grey line would move. It does not.

This is not a bug to be fixed before publishing; it is the result. Any closed red/blue loop
whose red team is synthetic will produce a curve like the one above, and without a
real-fraud control line in the same file there is no way to tell that curve apart from a
real advance. The curve is unfalsifiable without the control, so the control ships beside
the headline in `results/summary.json` and `results/curve.csv`.

**What still does not explain it: stealth.** Narrowed to comparable typologies, real fraud
spans 0.127–0.456 in distance-from-legit and is caught 0.2–23.1% of the time; the six
vectors span 0.270–0.483 — overlapping range — and are caught 0.00% of the time. Being
close to legitimate traffic is not what makes these attacks invisible. Being invented is.
See [`scripts/audit/narrowness_matched_ks.py`](scripts/audit/narrowness_matched_ks.py).

---

## Quickstart

**Python 3.10–3.12**, and that window is narrow for a reason worth knowing before you
hit it: `numpy==1.26.4` publishes no wheel for 3.13 or 3.14, and `pyarrow==25.0.1`
publishes none for 3.9 — which is what a bare `python3` still is on macOS. Declared in
`pyproject.toml` and `.python-version` so the failure is a clear message rather than a
compiler error that looks like this project's fault.

macOS prerequisite: LightGBM's wheel dynamically links Homebrew's OpenMP at import
time (`brew install libomp`) — without it `import lightgbm` fails even though pip
installed cleanly.

```bash
python3.12 -m venv .venv
.venv\Scripts\activate           # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

python scripts/prepare_ieee.py        # ONCE: real IEEE-CIS transactions -> derived features
                                      # idempotent; --force rebuilds. NOT optional for the
                                      # scripts below: six of the seven pin source="ieee"
                                      # and will refuse to run on the synthetic fallback.

python scripts/run_loop.py --fast     # 4-iteration smoke run
python scripts/run_loop.py            # full 8-iteration run -> results/  (~78s on real data)

python scripts/generalisation_check.py  # leave-one-vector-out: recall on an UNSEEN family
python scripts/mitigation_report.py     # score -> action -> money
python scripts/coordination_check.py    # what each attack design choice is actually worth
python scripts/ensemble_check.py        # does a second detector arm earn its place?
python scripts/latency_check.py         # can it run inside an authorization?
python scripts/feature_ablation.py      # why the feature space looks the way it does
                                        # HEAVY: ~13 GB peak RSS, reads the raw 683 MB CSV

streamlit run dashboard/app.py        # the 3-panel live demo (replays results/)
python -m pytest tests/ -q            # 84 tests: contract, leakage, optimizer, loop,
                                      # fidelity, mitigation, operating points
```

The dashboard **replays precomputed results** — it never trains live, so it can't stall
on stage.

---

## The loop interface contract (the day-1 lock)

Two frozen structs let the red and blue sides build in parallel — see
[`chhal/contract.py`](chhal/contract.py):

- **`AttackBatch`** — what the red team emits: fraud rows in exactly `FEATURE_COLUMNS`.
- **`ScoreReport`** — what the detector returns: precision / recall / F1 / AUC, FP rate on
  legit, and per-vector recall.

**The one rule that keeps the loop honest:** an attack may only contain columns the
detector can also see for legitimate traffic. No attack may invent a feature the detector
can't observe — otherwise "detection" is trivial and the result is fake. That single
constraint forces the arms race to happen in a realistic space.

---

## The attack vectors

Two tracks, kept explicitly separate (fudging this loses feasibility points):

**★ Live-loop vectors** (emit transaction features, flow through the detector, *are* the loop):

| Vector | Idea | Campaign shape | recall @ 0.1% FPR, round 8 |
|---|---|---|---|
| `threshold_hugging` | Per-victim mimicry: the campaign is sized, paced **and clocked** from the compromised card's **own** history, so nothing about it is anomalous *for that account* | 3-9 txns at the victim's own cadence, at the victim's own hours, spending the victim's own amounts | 58.2% |
| `autopay_mandate` | The inverse of a burst: a recurring-mandate hijack that pays monthly, flat, and trips no velocity feature | 4-8 txns at monthly cadence, near-constant amount | 63.2% |
| `mule_fanout` | One operator, many mule accounts, one window — the vector GenAI actually changes, because what it makes cheap is *scale* | 2-5 transfers per account, every account firing inside the same 6 hours | 87.8% |
| `upi_collect` | fraudulent UPI collect-request + rapid drain (India rail) | 3-7 hops, 30s-10min apart, each smaller as funds run out | 89.2% |
| `bustout` | GenAI synthetic identity ages a clean account, then busts out in a burst | quiet for days, then 8-25 txns minutes apart, amounts escalating 1.6× | 89.4% |
| `card_testing` | Agentic BIN/card probing sized to stay under velocity limits | 20-60 micro-probes, 2s-2min apart | 97.0% |

**Every one of these is 0.00% before the loop runs, and 0.00% after the optimizer runs on
the un-retrained detector.** The column above is what the detector reaches after eight
rounds of being retrained on this generator's output. Read it as a measurement of the
generator, not of the threat — see [What the arms-race curve actually
measures](#what-the-arms-race-curve-actually-measures).

Campaign shapes are not decoration — they are how the behavioural features are produced.
See [Attacks are campaigns, not rows](#attacks-are-campaigns-not-rows).

### What each vector is actually for — two of these are negative controls

Six vectors is not six equally interesting attacks, and saying so is cheaper than having a
reviewer say it. Two of them exist to **bound the scale**, not to evade anything:

**`card_testing` is the ceiling, and it is not novel.** Card testing is a named,
PCI-mandated fraud pattern that every acquirer already screens for; catching it at 97% is
table stakes, not a result. Its value here is threefold — it anchors the top of the
detectability scale, it is the loudest thing in the suite (KS 0.483 from legit, 43× the
noise floor), and it is the vector with a real-world confusable class, which is what makes
[the dunning control](#the-confusable-class--what-card_testings-97-actually-costs)
possible. **And it is the vector the detector generalises to worst**: held out of training
entirely it is reached only **12.2%** of the time, against 64.6% for `upi_collect` and
69.6% for `mule_fanout`. The loudest, most separable attack in the suite is the one a
detector trained on the other five cannot find. Whatever the detector learns about card
testing, it learns from card testing.

**`bustout` is the floor of the same scale, and it is not novel either.** Bust-out is an
industry-standard typology with a vendor literature behind it — TransUnion and Socure both
publish on it under "sleeper fraud" — and it is among the easiest vectors here (89.4%
caught). It is the control that says the detector is not broken: a loud, well-documented
attack pattern should be caught, and it is.

The other four carry the argument. `threshold_hugging` is per-victim mimicry (published,
[cited below](#two-of-these-do-something-the-other-three-do-not--and-one-ablation-that-did-not-go-our-way)),
`autopay_mandate` probes the opposite blind spot from every burst vector, `mule_fanout` is
a controlled experiment about a coordination signal the feature space cannot represent, and
`upi_collect` is a rail-specific variant.

### Six names, six populations — and that is the problem

An earlier version of this plan intended to demote `upi_collect` to a sibling of `bustout`,
on the reasoning that on the frozen 26 columns it is the same burst with the amount trend
inverted (0.7 draining against 1.6 escalating). **That is measurably false.** Train a
classifier to tell any two vectors apart on the ten columns the red team controls
(`scripts/audit/vector_separability.py`, out-of-fold AUC, n=800 each):

| | thr_hug | bustout | card_test | upi | mule | autopay |
|---|---|---|---|---|---|---|
| **threshold_hugging** | — | 0.997 | 1.000 | 0.996 | 0.989 | 0.987 |
| **bustout** | 0.997 | — | 1.000 | 0.997 | 0.998 | 1.000 |
| **card_testing** | 1.000 | 1.000 | — | 1.000 | 1.000 | 0.999 |
| **upi_collect** | 0.996 | 0.997 | 1.000 | — | 0.957 | 0.999 |
| **mule_fanout** | 0.989 | 0.998 | 1.000 | 0.957 | — | 0.996 |
| **autopay_mandate** | 0.987 | 1.000 | 0.999 | 0.999 | 0.996 | — |

`upi_collect` and `bustout` separate at **0.997**. The closest pair in the whole matrix is
`upi_collect` / `mule_fanout` at **0.957**, and even that is cleanly separable. So the
sibling framing is dropped rather than written — all six are distinct populations in the
feature space the detector actually sees.

**Which makes the real finding worse, not better.** Every pair here is separable at AUC
≥ 0.957 — a classifier trained for a few seconds tells any two of these apart almost
perfectly — and at a 0.1% false-positive budget the fraud detector catches **0.00%** of all
six. The vectors are not subtle. They carry loud, distinctive, trivially learnable
signatures. They are simply signatures the detector has no training data for, because they
were invented rather than observed. Separability and detectability are measuring different
things here, and only one of them is about fraud.

### Two of these do something the other three do not — and one ablation that did not go our way

**`threshold_hugging` is a published attack, and we claim the measurement, not the idea.**
Carminati, Polino, Continella, Lanzi, Maggi and Zanero, *Security Evaluation of a Banking
Fraud Analysis System*, **ACM Transactions on Privacy and Security 21(3), 2018** — mimicry
attacks against the Banksealer detector, evaluated on real data from a large Italian bank,
with the attacker's variables being exactly the transaction **amount** and **timestamp**.
<https://conand.me/publications/carminati-bankingfraud-2018.pdf>. The follow-up (RAID 2020,
<https://www.usenix.org/system/files/raid20-carminati.pdf>) states the architectural rule
this repo's `DERIVED_FEATURES` boundary implements — *"the features that can be fully
manipulated by the attacker are the amount and the timestamp"*, and an attacker *"needs to
inject in the banking system raw transactions (not aggregated ones)"*. What is ours is the
per-victim quantile calibration below, the fixed-FPR measurement, and the ablation that
prices it.

**`threshold_hugging` reads its bands off the victim, not off the population.** Every
vector's bands are quantile levels rather than raw values, which makes them portable. But
a quantile of *everyone's* traffic is a population-level disguise, and the crowd is not
what scores the transaction: the detector measures `amount_to_avg_ratio` and the
inter-transaction gap against **this card's** baseline. With `mimic_host`, the same
quantile levels are read off the host's own history, so a card that buys coffee gets a
coffee-sized attack at coffee-buying intervals.

**`mule_fanout` is a controlled experiment, not a fifth variation on speed and size.** Its
defining property — one operator moving many accounts at once — is a property of the
*set*, and the frozen feature space has no counterparty: no beneficiary id, no destination
account, no edge between two rows.

Both of those are the kind of claim that sounds obviously true, so
`scripts/coordination_check.py` makes them prove it: five seeds, five variants, three
operating points, each variant differing from its control in exactly one thing.

| switched on | costs the detector | read at | verdict |
|---|---|---|---|
| per-victim mimicry | **+2.3 ± 2.6** pts | 5% FPR | inside the noise |
| coordination | **−1.0 ± 0.6** pts | 5% FPR | inside the noise |
| `is_new_beneficiary` at all | **−25.0 ± 3.2** pts | 0.1% FPR | clears 2 SEM |
| …and hard-coding it to 1 | **−8.8 ± 4.9** pts | 0.1% FPR | inside the noise |

(± is the standard error of the paired per-seed difference, not the spread of one run. A
positive number is recall the detector *loses* when the choice is switched on, so it is a
gain for the attacker; a negative one means the choice makes the attack easier to catch.)

**The third row is the clean result, and it is not the one we set out to prove — it runs
against us.** The detector catches `mule_fanout` substantially on one binary column: turn
`is_new_beneficiary` off entirely and recall falls **25.0 ± 3.2** points at a 0.1% budget.
Some of that is legitimate — a mule fan-out really does send money somewhere new, and a
detector is entitled to use that.

**But "off entirely" is the wrong counterfactual, and asking the right one changes the
answer.** Zeroing the column on every row is itself a determinism the detector can read,
which is the same mistake the fourth row exists to price. Move the *rate* instead — from
the vector's 0.75 to the legitimate base rate of 0.552, still drawn per row — and the cost
is **+4.3 ± 1.3** points across three seeds (`scripts/mule_alpha_sweep.py`). So the vector
does **not** rest on that column: it rests on the column being *nearly always* set, and
that is a much smaller and much more defensible dependence.

The fourth row is the part that was ours. `is_new_beneficiary` used to be hard-coded to
**1 on 100% of rows** in four of the then-five vectors, and that determinism was worth a
further 8.8 ± 4.9 points on top of the genuine signal — which on the post-purge split no
longer clears its own error bars, so it is reported and not claimed. In real IEEE-CIS traffic the flag sits
on 55.2% of legitimate transactions and 42.2% of frauds — *less* common in real
fraud than in ordinary spending — so a vector that sets it every single time had painted
a target on itself. It is now a per-vector probability drawn from each vector's story
(0.30 for `threshold_hugging`, 0.75 for the mule, up to 0.90 for card testing), and the variant that
restores the old behaviour is kept in the ablation precisely so this row can exist.

**Coordination no longer clears its error bars, and an earlier version of this file said
it did.** On the pre-purge split it read +2.5 ± 0.8 points at a 1% budget and was written
up as clearing 2 SEM with a confound attached. On the split with the delay period and the
entity purge it reads **−1.0 ± 0.6** points — the wrong sign and inside the noise. The
confound was real and is now moot: a coordinated batch draws hosts that were live shortly
before its window, so the two mule variants never shared a host population, and this design
could not have separated timing from host mix even if the difference had held.

**Nor can the coordination be jittered away, and the reason is the point of the vector.**
`scripts/mule_alpha_sweep.py` sweeps an evasion budget α ∈ [0, 0.5] over the two operations
that have an analogue here. Widening the firing window from 6 to 30 hours moves recall
18.1% → 19.8% (inside the noise). Padding each mule with decoy transfers, 5 → 15 per
account, moves it 15.0% → 23.8% — the wrong way, because extra transactions add velocity
signal in a feature space that cannot see the network they belong to. The recipe comes from
**arXiv 2607.27370, which is Ethereum Sybil-cluster discovery via Gzip compression
distance, not a mule paper**; applying it here is a cross-domain translation and its third
operation, permutation, has no analogue at all, because there is no counterparty column and
therefore no graph to permute. That absence is what `mule_fanout` was built to measure,
arriving from the other direction.

**Per-victim mimicry does not survive its own ablation here.** 2.3 ± 2.6 points is not a
finding. `threshold_hugging` sits near the floor with mimicry and without it, so a single-pass
detector cannot tell them apart — the loop's per-vector recall does differ, but eight
rounds of adaptation is not a controlled comparison and we are not going to quote it as
one. The mechanism is still the right one on the merits; the evidence that it *matters* is
not there yet, and a larger seed budget is what would settle it.

**The thing this ablation actually taught us is about our own numbers.** An earlier
single-seed version of this script returned coordination deltas of −0.8, −3.2 and −8.3
points on three consecutive runs — same code, same data, different rng. Any one of them,
quoted alone, would have been a story invented from noise. That is a warning about every
per-vector number in this README quoted to one decimal place, and the reason
`scripts/robustness.py` exists.

### The confusable class — what `card_testing`'s 97% actually costs

`card_testing` is the easiest vector in the suite and the number is not impressive on its
own, because the cheapest way to reach it is to flag every repeated same-amount attempt on
a card. That is also what a **dunning run** looks like: when a subscription charge soft-
declines, the processor retries it on a ladder over the following days, same amount, same
merchant, low success rate. Stripe's own documentation warns that these "can look like card
testing".

So `chhal/redteam/vectors.py` carries a `Dunning` population — legitimate, `is_fraud = 0`,
deliberately **not** in `ALL_VECTORS`, never optimized against the detector, never trained
on. `scripts/dunning_control.py` scores it at the same 0.1% threshold, after three rounds
of adaptation on the full attack suite, over three seeds:

| population | | flagged at a 0.1% budget |
|---|---|---|
| `card_testing` | attack | **83.4% ± 0.3** caught |
| dunning, payee record reset | legit | **38.8% ± 6.6** false positive |
| dunning, known payee | legit | 1.5% ± 0.4 false positive |
| ordinary IEEE-CIS legit traffic | legit | 0.100% |

Read the middle rows. A detector at this operating point declines **1.5%** of a merchant's
subscription retries in the ordinary case — fifteen times the base rate — and **38.8%** of
them whenever a card update has reset the payee record, which is 389× the base rate and an
outage in everything but name.

And the gap between those two rows is the finding: they differ by a factor of **25**, and
the only thing that differs between them is `is_new_beneficiary`. **The detector is not
separating card testing from dunning by the shape of the retry sequence. It is reading who
is being paid.** That is the same single-column dependence the `mule_fanout` ablation found,
in a second vector, measured a second way — and it is the reason a card-testing recall
number should never be quoted without this table beside it.

---

## The constrained evasion optimizer — the novel core

Nudges attacker-controllable features to lower the detector's fraud score, but only within
a realistic, executable envelope. Without that guardrail you get attacks that fool the
model yet that no real fraudster could execute — which would destroy the feasibility
score. Every candidate must obey **(a)** business rules (velocity caps, valid amounts),
**(b)** the realistic manifold (feature quantile bounds from real data), and **(c)**
attacker control (issuer-side signals like `merchant_risk` are off-limits). See
[`chhal/optimizer.py`](chhal/optimizer.py).

---

## What is not ours

Six things in this repo look like contributions and are not. Naming them here is cheaper
than having a reviewer name them for us, and it makes the short list of what *is* ours
legible.

| Borrowed | From |
|---|---|
| **Measuring recall at a fixed false-positive budget** rather than at a 0.5 cutoff | Le Borgne, Siblini, Lebichot & Bontempi, *Reproducible Machine Learning for Credit Card Fraud Detection*, Ch. 4 — the standard protocol for a 3.5%-prevalence problem |
| **CP@k / precision-at-k as the operational metric** | Dal Pozzolo et al., ULB–Worldline; it is how a real alert queue is scored |
| **Anchoring a distance statistic to its own noise floor** | Sajja, arXiv:2604.13125 (Apr 2026) |
| **Leave-one-attack-out generalisation testing** | standard practice in the intrusion-detection literature, decades old |
| **The immutable-vs-attacker-controlled feature split** | Carminati et al., RAID 2020 — *"the features that can be fully manipulated by the attacker are the amount and the timestamp"* |
| **The closed red-team / blue-team retraining loop itself** | a commodity pattern by 2026; the interesting question is no longer whether you can build one |
| **Per-victim mimicry as an attack idea** | Carminati et al., ACM TOPS 21(3) 2018 |

What is ours is narrower and, we think, more useful: the **measurement** that both ends of
the arms-race curve track the generator rather than the threat, with the real-fraud control
line sitting in the same CSV as the headline; the **entity-level purge** that showed a
quarter of the detector's apparent skill was memorised accounts; and the **finding** that
distance-from-legit does not predict detection across provenance boundaries.

---

## Fidelity of simulation — measured, not claimed

A judged criterion, so we quantify it ([`chhal/fidelity.py`](chhal/fidelity.py)):

- **On-manifold rate 100.00%** — by construction, and scoped to the four features the
  attacker directly sets, which is all the guardrail governs. Over **all 26** features it
  is **98.17%**, and only **67.24%** of rows are fully inside the envelope on every
  column. All three are reported, because quoting the first alone invites the obvious
  question of what the other twenty-two are doing. (The answer is that the behavioural
  block is *derived from a timeline*, not assigned, so clipping it would mean forbidding
  an attack from having the shape it really has — a card-testing run that probes forty
  times in ten minutes HAS a velocity no legitimate account shows.)
- **Guardrail binding rate 33.44%** — the non-tautological number: the fraction of proposed
  perturbations that actually landed *outside* the manifold before clipping and had to be
  pulled back. This is the real evidence the guardrail does work — attacks push against
  the plausibility envelope, they don't just float freely inside it.
- **Per-vector KS distance from legit traffic**, reported two ways:

| vector | KS over all 26 features | KS over the 10 the red team controls | × the noise floor |
|---|---|---|---|
| `autopay_mandate` | 0.158 | 0.270 | 16.1× |
| `threshold_hugging` | 0.175 | 0.284 | 15.8× |
| `mule_fanout` | 0.208 | 0.335 | 31.9× |
| `upi_collect` | 0.215 | 0.381 | 25.6× |
| `bustout` | 0.232 | 0.410 | 34.7× |
| `card_testing` | 0.270 | 0.483 | 43.1× |

  The second column exists because the first one flatters us and it is worth saying why:
  16 of the 26 features are inherited whole from a real host account, so they match
  legitimate traffic by construction and each contributes a KS near zero. Averaging them
  in is 62% free passes.

  The third column exists because a raw KS statistic has no zero. Two 500-row samples of
  the *same* legitimate traffic score 0.024 apart on average, purely from sampling noise,
  so "KS = 0.15" means nothing until it is divided by that floor. Rescaled per feature and
  per sample size, two legit draws land at **1.4×** and the six vectors at **16–43×**. On
  the ten columns the red team actually sets, **one feature out of sixty** falls within
  twice the noise floor of legitimate traffic.

  **The ordering claim that used to sit here — closest to legit therefore hardest to
  catch — is withdrawn.** It does not survive the measurement: at 0.1% FPR every vector in
  this table is caught **0.00%** of the time, from 0.270 to 0.483, so distance from legit
  is not what determines detection here. Narrowed to comparable typologies, real fraud
  spans **0.127–0.456** on the same metric — overlapping this table almost exactly — and is
  caught **0.2–23.1%** of the time. Same distance band, opposite outcome. What separates
  them is not stealth; it is that one population is in the training distribution and the
  other was invented. See [What the arms-race curve actually
  measures](#what-the-arms-race-curve-actually-measures) and
  [`scripts/audit/narrowness_matched_ks.py`](scripts/audit/narrowness_matched_ks.py).

`results/fidelity.png` overlays the mimicry vector on a 6,000-row sample of the 73,156
real legitimate test transactions. Every distance on this page is a distance from real
legitimate traffic — see [Data](#data--real-not-invented).

## Mitigation — detect, flag, **and mitigate**

Detection stops at a probability. `score >= 0.5` is not a mitigation, and no payments
system is tuned that way. [`chhal/mitigation.py`](chhal/mitigation.py) picks, per
transaction, the action with the lowest **expected cost**:

| action | when it is fraud | when it is a real customer |
|---|---|---|
| `allow` | you eat the amount + chargeback fee | free |
| `step_up` | 90% cannot complete the OTP; the challenge fee is paid anyway | the fee, and 5% abandon the purchase |
| `review` | analyst catches 95% | analyst catches 95%, costs minutes and delay |
| `block` | no loss | lost margin **and** lasting goodwill |

Two things follow, and both are the point. The decision becomes **amount-aware** for
free — at p=0.30 a $5 transaction prices into a cheap OTP challenge (expected cost 4.16
vs 9.00 to allow) while a $5,000 one prices into analyst review (116.60 vs 188.98 to
challenge) — which no single global threshold can express. And both interventions are
rationed to a **capacity cap**: the review queue to 0.5% of traffic, because a policy
that routes 8% to analysts is not deployable however good its economics look, and
step-up to 5%, because an earlier version of this module made that argument about
analysts while itself challenging **15.2%** of all traffic and **14.3%** of legitimate
customers. A challenge is cheaper than an analyst, not free — every one of them is a
real customer stopped mid-payment — so it gets a budget too. Under the cap the policy
challenges **4.90%** of legitimate customers and declines **0.76%** outright.

It only works on **calibrated** probabilities. A raw gradient-boosting score is not
P(fraud), and this pool has attacks injected so its implied base rate is not the
deployment base rate either. An isotonic calibrator is fitted on a temporal slice of the
training window the detector never sees. That pool is then split in half — isotonic is
**fitted** on one half and its error is **measured** on the other, because an isotonic fit
scored on its own rows returns ECE 0.0000 by construction and that is a tautology, not a
result. Measured honestly: ECE 0.0070 raw → 0.0069 calibrated on the held-back half (the
raw LightGBM score is already close to calibrated here; what calibration buys is the
*scale*, which is what gets multiplied by a dollar amount). `test_miscalibrated_scores_degrade_the_policy`
locks this in: squash the scores monotonically — leaving every recall and AUC number
identical — and the policy measurably loses money.

```bash
python scripts/mitigation_report.py
```

Priced on the frozen future (76,617 real transactions + 2,400 adaptive attacks the
detector never saw):

| policy | cost per 1k txns | net cost reduction | fraud loss avoided |
|---|---|---|---|
| do nothing | $14,856.53 | — | — |
| block at `score >= 0.5` (untuned) | $7,658.24 | 48.45% | 52.99% |
| tuned allow/step-up/block (amount-blind) | $5,967.39 | 59.83% | 63.62% |
| expected-cost policy, analyst queue closed | $5,450.04 | 63.32% | 71.12% |
| **expected-cost policy** | **$5,216.67** | **64.89%** | **72.15%** |

**Two columns, because they are two different questions.** *Net cost reduction* nets the
friction we impose on legitimate customers against the fraud we stop — the number a CFO
signs off on. *Fraud loss avoided* ignores that friction and answers only how much of the
money that would have walked out we kept. An earlier version of this table reported the
first and labelled it the second.

**And five rows, because two of them would have been a straw man and one is the
attribution.** Beating `score >= 0.5` is not evidence that expected-cost decisions work;
it is evidence that 0.5 is a bad threshold, and nobody deploys an untuned one. The honest
comparator is the best *amount-blind* ladder that exists — allow below one threshold,
challenge between, block above — with both thresholds tuned on the same cost model, on a
slice held back from the one it is priced against. `tune_two_thresholds` finds that pair
exactly rather than by grid search (sorting by score turns it into prefix sums), so the
comparator is the strongest one available, not a convenient one.

It is also held to **the same friction budget our own policy has to live inside**, which
it was not before. An unconstrained ladder buys its low cost by challenging a large share of
all traffic — a number no issuer would ship — so comparing a 5%-capped policy against it
was rigged in one direction while the uncapped comparison was rigged in the other. (The
ladder's thresholds are tuned to spend at most 5% on the calibration half; applied to the
evaluation population they realise **5.69%**, because a fixed threshold does not carry a
fixed rate across populations. The comparator therefore spends *more* friction than our
policy is allowed to, so the edge below is a conservative one.)

Against it, **amount-awareness plus the analyst queue are worth 5.06 points of net cost
reduction** — $750.72 per thousand transactions, a 12.58% relative saving. The fourth row
splits that: with the queue closed the policy has the ladder's exact action set and the
ladder's exact friction budget, so the only thing left between them is pricing each
decision against the amount. Attributing the whole saving against the untuned cutoff:

| source | saving |
|---|---|
| tuning the thresholds | $133,606 |
| pricing against the amount | $40,879 |
| the analyst queue (0.5% of traffic) | $18,441 |

Tuning is still the biggest single piece, and any fraud team already does it.
`test_the_real_edge_is_measured_against_the_tuned_ladder_not_the_naive_threshold` asserts
this decomposition, including that the analyst queue must never be the largest term — if
it were, the honest headline would be about staffing an ops team, not about economics.

The policy declines **0.76%** of real customers outright, against 0.92% for the fixed
threshold, and challenges a further 4.90%, while stopping or challenging 58.9% of all
fraud (41.5% of real fraud, 83.9% of adaptive attacks). Note that the decline rates rose
once the linkage block was added: the detector is genuinely more confident on real fraud,
so blocking becomes economically correct more often. That is the policy working, but it
is a real cost and the cost model is where to argue about it.

### Whose fraud is being avoided

Detection is reported per segment; the economics was not, and that hid something. A
quarter of the cost denominator is fraud **we generated**, and the policy is far better at
our own attacks than at the real thing:

| segment | n | do nothing | expected-cost policy | net cost reduction |
|---|---|---|---|---|
| **real fraud + all legitimate traffic** | 76,617 | $638,020 | $371,361 | **41.79%** |
| adaptive attacks only | 2,400 | $535,899 | $40,845 | 92.38% |
| real fraud only, excluding friction on legit | 3,461 | $638,020 | $286,142 | 55.15% |

The blended 64.89% borrows credit from attacks we wrote ourselves. **41.79% is the number
that would survive contact with a production book**, and it is the one to quote. The third
row is the same book with the cost of challenging and declining real customers taken out —
useful for seeing where the money goes, but it is not a number to quote on its own, because
that friction is real and somebody pays it.

**Everything above this table is a second paper, and it is fenced off deliberately.** The
cost model rests on constants nobody measured here — a 90% OTP failure rate for fraudsters,
a 95% analyst catch rate, a dollar price on lost goodwill — and every headline in it moves
when those move. This segment table is the exception: it is a *ratio* between two
populations priced by the same model, so it survives any cost model that is applied
consistently to both. **92.38% on attacks we wrote, 41.79% on the real book.** That is
on-thesis and it is the only number from this section worth carrying forward.

### An honest split we are not hiding

Recall at 0.1% false positives on real legitimate traffic, by segment:

| segment | recall |
|---|---|
| unseen adaptive attacks (our red team) | **80.8%** |
| real IEEE-CIS fraud | **14.2%** |

Real fraud was **2.3%** before the linkage block was added — see
[Mounting attacks on real accounts](#mounting-attacks-on-real-accounts). It is now 14.2%,
a 6.3x lift, and the detector leans on those features heavily. That trade is visible rather
than hidden, and it is the right one for a submission that has to work on real payments.

**These two rows are the whole paper in miniature.** 80.8% is what eight rounds of
adversarial retraining achieve against a red team we wrote. 14.2% is what the same detector
does to fraud that actually happened, and it is the number that does not move no matter how
many rounds are run.

Reported once and not led with: **CP@100 is 31.8% ± 8.6** on the same split
(`scripts/audit/card_precision_at_k.py`), 7.4× a randomly filled queue. Card Precision at k
is the closest thing the fraud literature has to an operational metric — rank cards by their
highest score of the day, work the top hundred — and it is a far friendlier number than
14.2% because ranking a hundred cards is a much easier problem than separating fraud from
73,000 transactions at a fixed false-positive rate. Both are true. Leading with the
friendlier one, in a repo whose finding is that this system has been measuring itself too
kindly, would hand a reviewer their paragraph.

These are thresholded on the **raw** detector score, not the calibrated one. Calibration
is monotone so it cannot change the ranking — but isotonic collapses long runs of scores
onto a single value, and at a 0.1% budget the threshold lands inside such a plateau, where
a tie-break decides whether hundreds of rows count as caught. Detection is a property of
the ranking; the calibrated probability is only needed for the economics.

Reporting the blended 33.9% would hide both halves. The mitigation layer closes more of
the remaining gap — most of the rest gets stopped or challenged, because a cheap OTP
is worth issuing on a weak signal even when an outright decline is not.

## A second arm — and an honest negative result

The supervised detector has a blind spot no amount of retraining removes: it can only
recognise what it has been shown, and the brief is about *emerging* attacks. So we added
an **anomaly arm** — an isolation forest trained on legitimate traffic only, which never
sees a fraud label and answers a different question: how unlike normal traffic is this?

Then we measured it (`scripts/ensemble_check.py`), leave-one-vector-out, at a fixed 0.1%
false-positive budget:

| variant | unseen attack family | real IEEE-CIS fraud |
|---|---|---|
| supervised only | **0.5444** | **0.1498** |
| max fusion (either arm flags) | 0.4644 ✗ | 0.1114 ✗ |
| stacked (anomaly score as a feature) | 0.5200 ✗ | 0.1383 ✗ |

**The anomaly arm alone now scores 0.0000 on unseen attacks and 0.006 on real fraud**, and
carries **0.0%** of the catches — not a rounding of something small, but nothing at all.
`max` fusion loses badly, 8.0 points on unseen families; stacking loses 2.4.

**Do not read the sign of any single run as a result.** Across five measurements the
stacking delta on unseen families came out at −1.25, −3.20, +2.00, −0.40 and −2.44 points
— it has changed sign between runs of the same code. A component whose entire case is a two-point swing that
flips sign has not earned 32MB of the 34MB model footprint, and "it won this time" is
exactly the reasoning this README exists to avoid. The script prints whichever way the
current run fell; the decision not to ship it is made on the fact that the number is not
stable, not on the sign of any one run. It also costs 32MB of the 34MB model footprint.

**This verdict has now changed twice, and that is the point.** On the earlier
twelve-feature space, stacking was worth +1.25 points and we shipped it. Once the linkage
block arrived and real-fraud recall went from 2.3% to 14.2%, the supervised arm had enough
signal that an outlier score added nothing but false positives. Adding a fifth vector
moved it back across zero. The module docstring did not keep up
and kept recommending stacking for several commits after the script had stopped agreeing;
the recommendation is now derived from the two numbers rather than written beside them. A component that earns its place at one
stage of a project can stop earning it at the next, and the only way to know is to keep
re-running the measurement rather than trusting the decision that was once correct.
**Neither fusion is used by default now.** Both remain in the repo so the result stays
reproducible.

The reason is our own doing, and it is the interesting part. Attack rows are drawn
through the inverse CDF of real legitimate traffic, clipped to the plausibility manifold,
and now mounted on real accounts whose issuer-side context they inherit outright — they
are on-manifold **by construction**, because that is precisely what the fidelity work
guarantees. A detector whose whole question is "is this off-manifold?" cannot see them.
**The better the fidelity claim gets, the less an outlier detector can contribute**, and
the numbers above moved exactly that way. `test_anomaly_arm_is_blind_to_on_manifold_attacks`
locks it in.

`StackedDetector` exposes the same surface as `Detector`, so the loop, evaluation and the
mitigation policy take it unchanged. The anomaly score is an issuer-side signal computed
at scoring time, not something an attacker sets, so `FEATURE_COLUMNS` stays frozen.

## Mounting attacks on real accounts

The twelve hand-derived features caught **3.0%** of real IEEE-CIS fraud at a 0.1%
false-positive budget. Adding the dataset's anonymised entity-linkage counts (`C1-C14` —
how many addresses, devices, emails and cards associate with this card) takes that to
**15.5%**. `scripts/feature_ablation.py` (heavy: ~13 GB peak RSS, ~45s, and it reads the
raw 683 MB CSV rather than the prepared parquet), on the same split as everything else —
temporal cut, 7-day delay period, straddling accounts purged:

| features | n | recall @ 0.1% FPR | PR AUC |
|---|---|---|---|
| the 12 we hand-derived | 12 | 2.25% | 0.203 |
| + linkage counts **we built ourselves** | 20 | 2.02% | 0.207 |
| **+ the dataset's C1-C14** | 26 | **15.46%** | **0.499** |
| + both | 34 | 15.86% | 0.502 |
| + C1-C14 and D1-D15 | 41 | 17.25% | 0.516 |
| + everything incl. all 339 V-features | 388 | 17.19% | 0.523 |

Three things fall out of that table. The linkage block is the entire story — all 339
V-features add **nothing** on top of it at this budget (17.19% against 17.25% for 41
features), and `D1-D15` are worth under two points. We tried to rebuild the same signal
from what we *do* understand (distinct counterparties, addresses, emails and card
attributes per account over time, plus longer velocity windows) and recovered
**−0.23 points** — the reconstruction is slightly *worse* than not trying. And putting ours
on top of theirs adds **+0.40**, inside the run-to-run spread. They are subsumed, not
merely weaker. Whatever C1-C14 aggregate over lives in devices, phones, IPs and cross-card
relationships this dataset does not expose.

This table also moved when the split did. An earlier version read 3.02% → 19.41%, measured
before the delay period and the straddler purge; the same experiment on the clean split
reads 2.25% → 15.46%. The *shape* of the finding is unchanged and that is the point of
reporting both — the linkage block still does all the work, it just does less of it than
the leaky split suggested.

### So the red team compromises a real card

A red team does not invent its victims. Each campaign is now mounted on a **real,
never-fraudulent account** ([`chhal/redteam/hosts.py`](chhal/redteam/hosts.py)). Its
linkage history, its age, the issuer's opinion of its merchants are whatever they
actually were, because they belong to an account that really existed in the data. The
story is the true one: an ordinary customer's card was taken over.

The feature space partitions cleanly, and the partition is the design:

| | count | who sets it |
|---|---|---|
| **attacker-controlled** | 9 | the fraudster: amount, timing, velocity, payee, rail, destination |
| **inherited** | 16 | the issuer: account age, merchant risk, the 14 linkage counts |
| derived from the timeline | 1 | `day_of_week` |

The evasion optimizer moves only the first group, and a test asserts it leaves the second
byte-identical. It also **only clips the first group** to the plausibility manifold: a
value a real account actually held is plausible by definition, and clipping it to a
q0.5%/q99.5% envelope would silently rewrite the issuer's own view of the card.

### Leakage rules, all enforced in code

- Only accounts whose every observed transaction is legitimate may host a campaign — a
  fraudulent account's rows carry label information.
- **The base split carries a 7-day delay period.** Before this, the gap between the last
  train transaction and the first test transaction was **60 seconds**. A model trained
  right up to the instant it is evaluated is not being asked the question a deployed model
  faces, which is what it knows about tomorrow. 18,550 rows are held out to create the gap.
- **Straddling accounts are purged from test entirely.** 52,468 of 129,085 post-embargo
  test rows — **40.6%**, across 21,337 accounts — sat on accounts the detector also trains
  on. A temporal split alone does not prevent this: an account that transacts on both sides
  of the cut lands in both halves, and every behavioural feature here is computed *within*
  an account. Those rows are not unseen customers; they are memorised ones, scored a little
  later. **Removing them costs 25.5% of the detector's measured skill on real fraud**
  (19.10% → 14.22% recall at a 0.1% budget) — which is the leakage measurement, and the
  reason it is published rather than fixed quietly.
- Evaluation pools additionally exclude any account seen in training when a campaign is
  mounted. (Belt and braces: hosts are all-legitimate, so recognising one would push an
  attack toward *legit* and make detection harder, not easier.)
- **`merchant_risk` is encoded causally.** It is a smoothed historical fraud rate per
  merchant bucket, and it used to be computed with a random 5-fold out-of-fold scheme.
  That stops a row encoding *itself*, which is the leak everyone checks for, and on
  temporally ordered data it leaves a second one open: the other four folds span the whole
  train window, so a January transaction was scored with a merchant risk number built
  partly from June. It is now an expanding window over transaction time — each row sees
  only what had already happened in its bucket — with simultaneous rows excluded as a
  block. Fixing it cost **nothing measurable** (real-fraud recall 14.22% before and after),
  which is worth stating: the detector was not leaning on that leak, so the 25.5% above is
  the entity purge alone.
- The train/held-out split is by **campaign**, not by row. A campaign is one compromised
  account and its 3-60 transactions, every one of them carrying that account's age,
  merchant history and fourteen linkage counts — so splitting rows at random left 98.1%
  of the "never trained on" rows sharing a host with a row the detector had just learned.
- None of the above is asserted in a docstring any more. `run_loop` computes a leakage
  audit each run and writes it into `results/summary.json`: benchmark rows found in the
  training pool, pressure rows found in the training pool, and benchmark host accounts
  that also appear in train. All three are **0**, and a reader can check that rather
  than take this section's word for it.
- Attack transactions are timestamped strictly **after** the host's last real
  transaction. A campaign continues an account; it cannot reach into its past.
- Inherited values are read from the host's last real transaction — the most recent state
  anyone could legitimately know at the moment of takeover.

### What it cost and what it bought

| | before | after |
|---|---|---|
| real IEEE-CIS fraud, recall @ 0.1% FPR | 2.3% | **14.2%** |
| unseen adaptive attacks | 98.8% | 80.8% |
| mimicry vector, KS distance from legit | 0.361 | **0.175** |

Fidelity improved sharply because sixteen of twenty-six features are now literally real
values. Adaptive-attack recall paid 25 points for it, which is the honest consequence of
the detector having real signal to lean on rather than an artefact of our own generator.
The policy's economics are not in this table on purpose: the "after" side is measured
under a corrected definition (net cost reduction, against a tuned comparator), so a
before/after row would be comparing two different quantities.

### The limitation we are not hiding

Linkage counts are **frozen** at the host's last observed values. A real takeover would
nudge some of them — a new shipping address raises whatever counts addresses. We cannot
model that, because we do not know what each column counts. Freezing is the conservative
choice: it means the detector **cannot** use linkage to catch our attacks, only to catch
real fraud. That is the correct behaviour for a feature the attacker does not control,
and it is why adding this block did not inflate our own numbers.

---

## Attacks are campaigns, not rows

`velocity_1h`, `velocity_24h`, `time_since_last_txn_min` and `amount_to_avg_ratio` are
four views of one timeline. Sampling them independently produces transactions that cannot
exist — and it did:

| | violates the 1h rule | 1h count > 24h count |
|---|---|---|
| `threshold_hugging`, before | 69.9% | — |
| every vector after `render()`, but **through the old optimizer** | 59.2 – 93.2% | 13.4 – 42.4% |
| real IEEE-CIS traffic | 0% | 0% |
| **every vector, through the optimizer, now** | **0%** | **0%** |

*The rule: if k transactions happened in the last hour, the previous one was at most an
hour ago. And a one-hour count cannot exceed the twenty-four-hour window containing it.*

The second row is the one worth dwelling on. Deriving behaviour from a timeline fixed the
renderer, and for a while that was reported as the whole fix — because the consistency
check ran on the seed batch. But the evasion optimizer then perturbed `velocity_1h`,
`velocity_24h` and `time_since_last_txn_min` as three independent scalars, and its output
is what becomes the benchmark, the fidelity population and the rows added to training. So
nothing downstream was ever measured on a coherent transaction, and the suite stayed green
throughout, because no test looked at the optimizer's output.

The optimizer now searches over the timeline itself — when to transact and for how much —
and re-derives the rest through the same `chhal.behaviour.derive` applied to all 590,540
real transactions. `tests/test_contract.py::test_the_optimizer_cannot_emit_an_impossible_transaction`
asserts it on optimized output for every vector, and `results/summary.json` reports it on
the shipped benchmark. This is measured, not argued.

So a vector now declares a `TemporalProfile` — how many accounts, how many transactions
each, how far apart, how the amount moves — and the generator lays out an actual
timeline. `card_testing` fires 20-60 probes seconds apart; `bustout` ages quietly for
days then bursts over a few hours with escalating amounts; `upi_collect` drains through
3-7 hops minutes apart, each smaller than the last; `threshold_hugging` moves at an
ordinary cadence of an hour to two days.

The behavioural features are then **derived** from that timeline by
[`chhal/behaviour.py`](chhal/behaviour.py) — the same function `scripts/prepare_ieee.py`
calls on the 590,540 real transactions. One implementation, both sides: whatever
relationships hold between these features in real data hold in the attacks, because it is
literally the same arithmetic. Consistency is not enforced afterwards, it is impossible to
violate. `hour` and `day_of_week` come from the timestamps too.

The history `amount_to_avg_ratio` is measured against is not generated either: it is the
host account's **real** transaction history. A bust-out reads as anomalous against what
that card actually spent, which is the entire signal.

Account-level features (`account_age_days`, `is_cross_border`, `channel_code`) are
sampled once per account and broadcast: one card does not change country or rail partway
through a campaign.

---

## Latency — can this run inside an authorization?

A card authorization is a synchronous round trip with a budget of roughly 100-300ms, most
of it network and issuer systems. The risk decision gets tens of milliseconds. So the
number that matters is the **full path at n=1** — anomaly score, detector, calibration,
action decision — one transaction at a time, which is how authorizations actually arrive.

```bash
python scripts/latency_check.py
```

| | p50 | p95 | p99 |
|---|---|---|---|
| **full path, single transaction** | **1.26 ms** | 1.35 ms | **1.42 ms** |

**~35× headroom** against a 50ms risk-decision budget at p99. Batch throughput is
**172,740 txns/sec** (5.8 µs each) at a batch of 10,000 — that is the nightly-rescoring
number, not the live-auth one, and should not be quoted as such.

Model footprint is 34MB, of which the anomaly arm is 32MB — dropping it, which the
measurement above says to do anyway, leaves under 2MB. There are no external lookups, no
feature store and no network calls on the scoring path.

## Repository layout

```
chhal/
  contract.py      # AttackBatch, ScoreReport, FEATURE_COLUMNS — the frozen interface
  data.py          # real IEEE-CIS base population (synthetic fallback); temporal split
  redteam/hosts.py # real accounts a campaign may compromise, and the leakage rules
  detector.py      # LightGBM blue-team detector (gain-based feature importance)
  redteam/         # the six live-loop attack vectors + the dunning negative control
  optimizer.py     # constrained evasion optimizer (the novel core)
  evaluation.py    # held-out split protocol + metrics
  fidelity.py      # KS-tests + on-manifold rate — fidelity as a metric, not a claim
  behaviour.py     # timeline -> velocity/recency/ratio; used on real data AND attacks
  mitigation.py    # calibration + expected-cost action policy — the "mitigate" pillar
  ensemble.py      # anomaly arm + StackedDetector; see the negative result above
  redteam/campaign.py  # TemporalProfile — how each vector unfolds on an account
  loop.py          # orchestration -> the arms-race curve
scripts/prepare_ieee.py          # one-time: raw IEEE-CIS -> derived FEATURE_COLUMNS
scripts/feature_ablation.py      # which features carry the real-fraud signal, and why
scripts/run_loop.py              # run the loop, write results/
scripts/generalisation_check.py  # leave-one-vector-out recall on an unseen attack family
scripts/mitigation_report.py     # calibrate, decide, price the policies
scripts/coordination_check.py    # ablate coordination vs the per-row tells
scripts/ensemble_check.py        # supervised vs max-fusion vs stacked, leave-one-out
scripts/latency_check.py         # per-transaction latency, throughput, footprint
dashboard/app.py                 # 3-panel Streamlit demo (replays results/)
tests/                           # contract, optimizer, loop, fidelity, mitigation
```

## Data — real, not invented

Every headline number is measured on **IEEE-CIS Fraud Detection** (Vesta Corporation):
**590,540 real card transactions over 182 days, 20,663 frauds (3.499%)**. Fidelity of
simulation is a judged criterion and it is judged against real payment data — a distance
measured against a distribution we invented ourselves would prove nothing.

```bash
python scripts/prepare_ieee.py     # downloads the real transactions, derives FEATURE_COLUMNS
```

[`scripts/prepare_ieee.py`](scripts/prepare_ieee.py) derives all twenty-six features from
raw IEEE-CIS: velocity, recency and amount-to-average are computed **within a reconstructed
account** (`card1 + addr1 + first-seen-day`, the community-standard uid) over the real time
ordering, using only transactions strictly before the row they describe. `account_age_days`
is the dataset's own `D1`; `is_cross_border` is `addr2 != 87`; `merchant_risk` is a smoothed
historical fraud rate fit **out-of-fold on the training split only**; the fourteen
entity-linkage counts are the dataset's own `C1-C14`, carried through unchanged and
inherited rather than generated (see
[Mounting attacks on real accounts](#mounting-attacks-on-real-accounts)). Every
approximation is documented in that file's header.

The split is **temporal** — the first 75% of the window trains, the last 25% tests. A random
split leaks future fraud patterns backwards and inflates every metric. The plausibility
manifold used by the evasion optimizer is computed on **train only**.

### The synthetic fallback, and why it is not the default

`load_base_data(source="synthetic")` keeps the original programmatic distribution so the repo
still runs end to end with no download. It cannot represent entity linkage at all — those
counts aggregate over devices and cross-card relationships no generator here has, so they
are zero-filled. It should never be quoted, and measuring it against real data shows why:

| feature | real IEEE-CIS (p50) | synthetic (p50) |
|---|---|---|
| `amount` | 68.50 | 892.28 |
| `velocity_24h` | 0 | 6 |
| `time_since_last_txn_min` | 22,966 | 166 |
| `account_age_days` | 3 | 694 |

The synthetic distribution claimed the median account transacts six times a day where the
real median transacts zero. Worse, its synthetic *fraud* multiplied amount by 2.5-6x,
inventing a separation that does not exist: in real data fraud averages 149.2 against
legitimate 134.5. Detection looked far easier than it is.

Runs record which population they used in `results/summary.json` under `data_source`.

## Team

Demilade · Akshat. See [`STRATEGY.md`](STRATEGY.md) for the full concept, taxonomy, and
scoring rationale.

*Metrics in the deck are goals to hit and measure; the held-out protocol is what lets us
report them credibly.*
