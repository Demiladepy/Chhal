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
   (4 GenAI attack           OPTIMIZER                  (LightGBM,
    vectors, tabular)     (evade + stay plausible)      flags fraud)
        ▲                                                         │
        │                                                         ▼
        └────────  held-out novel attacks feed back  ◄──── retrain + score
```

---

## The money chart: two honest lines

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

**No leakage:** the base split is temporal and frozen before any attack is injected; the
benchmark and pressure attacks touch neither.

### Measured where a payments team would actually run it

Not at `score >= 0.5`, and not in ROC AUC. Fraud systems are tuned to a **false-positive
budget** — flag no more than X% of good customers, catch as much as possible inside that —
because flagging good customers is the expensive failure. And at 3.5% prevalence ROC AUC is
flattered by an enormous true-negative pile: 0.9999 there is unremarkable. So the headline
is **recall at a fixed FPR** and **PR AUC** (average precision), with the 0.5-threshold
numbers kept only for comparison.

Full run, 8 iterations on real IEEE-CIS (`scripts/run_loop.py`, ~85s):

| metric | baseline | after the loop |
|---|---|---|
| recall @ **0.1%** of legit flagged | 0.00% | **98.90%** |
| recall @ 0.5% | 0.00% | 98.95% |
| recall @ 1.0% | 0.00% | 99.10% |
| **PR AUC** | 0.0072 | **0.9923** |
| alert rate (share of all traffic) | 0.10% | 1.467% |

For comparison, the naive 0.5 cutoff on the same run: F1 0.0 → 0.9656, ROC AUC 0.9987, FP on
legit 0.08%. The ROC number is the one to distrust.

The baseline catching **zero** is not a broken detector — the benchmark attacks were
optimised specifically to evade it, which is the optimizer doing its job. The claim that
survives scrutiny is not this curve but the leave-one-out one: **88.7% on an attack family
never seen in any form** (`scripts/generalisation_check.py`). Numbers vary by seed.

---

## Quickstart

macOS prerequisite: LightGBM's wheel dynamically links Homebrew's OpenMP at import
time (`brew install libomp`) — without it `import lightgbm` fails even though pip
installed cleanly.

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

python scripts/prepare_ieee.py        # ONCE: real IEEE-CIS transactions -> derived features
                                      # (skip it and everything falls back to synthetic)

python scripts/run_loop.py --fast     # ~1 min smoke run (4 iterations)
python scripts/run_loop.py            # full 8-iteration run -> results/  (~85s on real data)

python scripts/generalisation_check.py  # leave-one-vector-out: recall on an UNSEEN family
python scripts/mitigation_report.py     # score -> action -> money
python scripts/ensemble_check.py        # does a second detector arm earn its place?
python scripts/latency_check.py         # can it run inside an authorization?

streamlit run dashboard/app.py        # the 3-panel live demo (replays results/)
python -m pytest tests/ -q            # contract, optimizer, loop, fidelity, mitigation
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

| Vector | Idea | Campaign shape |
|---|---|---|
| `threshold_hugging` **(hero)** | LLM-tuned sequences that sit just under every velocity/amount rule and mimic the victim's normal behaviour — the hardest to catch, the heart of the arms race | 3-9 txns, 1h-2d apart, spend equal to the account's own normal |
| `bustout` | GenAI synthetic identity ages a clean account, then busts out in a burst | quiet for days, then 8-25 txns minutes apart, amounts escalating 1.6× |
| `card_testing` | Agentic BIN/card probing sized to stay under velocity limits | 20-60 micro-probes, 2s-2min apart, on a card with no history |
| `upi_collect` | 🇮🇳 fraudulent UPI collect-request + rapid drain (India rail) | 3-7 hops, 30s-10min apart, each smaller as funds run out |

Campaign shapes are not decoration — they are how the behavioural features are produced.
See [Attacks are campaigns, not rows](#attacks-are-campaigns-not-rows).

**◆ Showcase vectors** (text/agent/media — voice clone, prompt-injection against payment
agents): scored in the write-up for breadth and demoed qualitatively. They do **not** emit
tabular features, so they do **not** feed the tabular detector — and we say so plainly
rather than pretend they close the same loop.

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

## Fidelity of simulation — measured, not claimed

A judged criterion, so we quantify it ([`chhal/fidelity.py`](chhal/fidelity.py)):

- **On-manifold rate ~100%** — by construction: the optimizer hard-clips every candidate to
  these exact bounds before scoring, so this only proves the clip is wired correctly.
- **Guardrail binding rate ~29%** — the non-tautological number: the fraction of proposed
  perturbations that actually landed *outside* the manifold before clipping and had to be
  pulled back. This is the real evidence the guardrail does work — attacks push against
  the plausibility envelope, they don't just float freely inside it.
- **Per-vector KS distance from legit traffic** — the hero `threshold_hugging` sits **closest
  to legit** (KS ~0.27), which is exactly why it's the hardest to catch; overt vectors
  (`bustout`, `card_testing`) sit further out **by design** — that separation is the fraud
  signal, not a defect. The ranking lines up with detection recall: stealthier ⇒ harder.

`results/fidelity.png` overlays the mimicry vector on legitimate traffic. Base data is
generated; swap in real PaySim / IEEE-CIS and every number becomes a real-data report.

## Mitigation — detect, flag, **and mitigate**

Detection stops at a probability. `score >= 0.5` is not a mitigation, and no payments
system is tuned that way. [`chhal/mitigation.py`](chhal/mitigation.py) picks, per
transaction, the action with the lowest **expected cost**:

| action | when it is fraud | when it is a real customer |
|---|---|---|
| `allow` | you eat the amount + chargeback fee | free |
| `step_up` | 90% cannot complete the OTP | small fee, 5% abandon the purchase |
| `review` | analyst catches 95% | analyst catches 95%, costs minutes and delay |
| `block` | no loss | lost margin **and** lasting goodwill |

Two things follow, and both are the point. The decision becomes **amount-aware** for
free — at p=0.30 a $5 transaction prices into a cheap OTP challenge (expected cost 3.56
vs 9.00 to allow) while a $5,000 one prices into analyst review (116.60 vs 188.37 to
challenge) — which no single global threshold can express. And the review queue is
rationed to a **capacity cap**, because a policy that routes 8% of traffic to analysts
is not deployable however good its economics look.

It only works on **calibrated** probabilities. A raw gradient-boosting score is not
P(fraud), and this pool has attacks injected so its implied base rate is not the
deployment base rate either. An isotonic calibrator is fitted on a temporal slice of the
training window the detector never sees (ECE 0.0087 → 0.0000). `test_miscalibrated_scores_degrade_the_policy`
locks this in: squash the scores monotonically — leaving every recall and AUC number
identical — and the policy measurably loses money.

```bash
python scripts/mitigation_report.py
```

Priced on the frozen future (147,635 real transactions + 1,600 adaptive attacks the
detector never saw, 4.49% fraud):

| policy | cost per 1k txns | loss avoided |
|---|---|---|
| do nothing | $8,372.50 | — |
| block at `score >= 0.5` | $5,938.46 | 29.1% |
| **expected-cost policy** | **$3,155.72** | **62.3%** |

It declines **0.031%** of real customers outright, against 0.086% for the fixed
threshold, while stopping or challenging 76.5% of all fraud — because most of the work
is done by cheap OTP challenges (26.8% of traffic) rather than declines (1.1%).

### An honest split we are not hiding

Recall at 0.1% false positives on real legitimate traffic, by segment:

| segment | recall |
|---|---|
| unseen adaptive attacks (our red team) | **98.8%** |
| real IEEE-CIS fraud | **3.6%** |

The loop beats its own red team and barely detects ordinary card fraud. That is a
feature-space limit, not a modelling one: twelve hand-derived features were chosen to
carry the *attack* narrative, where IEEE-CIS leaderboard entries use several hundred
engineered ones. Reporting the blended 26.0% would hide both halves. The mitigation
layer is what partially rescues it — 69.3% of real fraud still gets stopped or
challenged, because a cheap OTP is worth issuing on a weak signal even when an outright
decline is not.

## A second arm — and an honest negative result

The supervised detector has a blind spot no amount of retraining removes: it can only
recognise what it has been shown, and the brief is about *emerging* attacks. So we added
an **anomaly arm** — an isolation forest trained on legitimate traffic only, which never
sees a fraud label and answers a different question: how unlike normal traffic is this?

Then we measured it (`scripts/ensemble_check.py`), leave-one-vector-out, at a fixed 0.1%
false-positive budget:

| variant | unseen attack family | real IEEE-CIS fraud |
|---|---|---|
| supervised only | 0.8855 | 0.0296 |
| **max fusion** (either arm flags) | **0.8735** ✗ | **0.0200** ✗ |
| **stacked** (anomaly score as a feature) | **0.8980** ✓ | 0.0287 |

**The anomaly arm alone scores 0.002 on unseen attacks and 0.000 on real fraud.** Fusing
it by `max` is actively worse than not having it — it spends part of the false-positive
budget on an arm that carries 0.2% of the catches. Only stacking its score as an input
*feature* helps, +1.25 points on an unseen family. It also costs 76.5MB of the 78MB total
model footprint and takes single-transaction latency from 0.26ms to 1.18ms, so whether
+1.25 points is worth it is a real decision, not an obvious one.

The reason is our own doing, and it is the interesting part. Attack rows are drawn
through the inverse CDF of real legitimate traffic and then clipped to the plausibility
manifold — they are on-manifold **by construction**, because that is precisely what the
fidelity guardrail guarantees. A detector whose whole question is "is this off-manifold?"
cannot see them. **The better the fidelity claim gets, the less an outlier detector can
contribute.** `test_anomaly_arm_is_blind_to_on_manifold_attacks` locks that in.

`StackedDetector` exposes the same surface as `Detector`, so the loop, evaluation and the
mitigation policy take it unchanged. The anomaly score is an issuer-side signal computed
at scoring time, not something an attacker sets, so `FEATURE_COLUMNS` stays frozen.

## Attacks are campaigns, not rows

`velocity_1h`, `velocity_24h`, `time_since_last_txn_min` and `amount_to_avg_ratio` are
four views of one timeline. Sampling them independently produces transactions that cannot
exist — and it did:

| | violates the 1h rule | violates the 24h rule |
|---|---|---|
| `threshold_hugging` (hero vector), before | 69.9% | **100%** |
| real IEEE-CIS traffic | 0% | 0% |
| every vector, now | **0%** | **0%** |

*The rule: if k transactions happened in the last hour, the previous one was at most an
hour ago.* The hero vector was physically impossible in every single row. Real traffic
never is, because it is derived from timelines.

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

Each campaign also carries a short history of ordinary spend before the attack begins.
That is what makes `amount_to_avg_ratio` mean *"large for this account"* rather than
just *"large"* — a bust-out reads as anomalous against what that card actually spent,
which is the entire signal.

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
| **full path, single transaction** | **1.18 ms** | 1.22 ms | **1.24 ms** |
| detector alone, single transaction | 0.26 ms | — | — |

**~40× headroom** against a 50ms risk-decision budget at p99. Batch throughput is
**138,907 txns/sec** (7.2 µs each) at a batch of 10,000 — that is the nightly-rescoring
number, not the live-auth one, and should not be quoted as such.

Model footprint is 78MB total, of which the anomaly arm is 76.5MB. There are no external
lookups, no feature store and no network calls on the scoring path.

## Repository layout

```
chhal/
  contract.py      # AttackBatch, ScoreReport, FEATURE_COLUMNS — the frozen interface
  data.py          # real IEEE-CIS base population (synthetic fallback); temporal split
  detector.py      # LightGBM blue-team detector (gain-based feature importance)
  redteam/         # the four live-loop attack vectors, calibrated to the real population
  optimizer.py     # constrained evasion optimizer (the novel core)
  evaluation.py    # held-out split protocol + metrics
  fidelity.py      # KS-tests + on-manifold rate — fidelity as a metric, not a claim
  behaviour.py     # timeline -> velocity/recency/ratio; used on real data AND attacks
  mitigation.py    # calibration + expected-cost action policy — the "mitigate" pillar
  ensemble.py      # anomaly arm + StackedDetector; see the negative result above
  redteam/campaign.py  # TemporalProfile — how each vector unfolds on an account
  loop.py          # orchestration -> the arms-race curve
scripts/prepare_ieee.py          # one-time: raw IEEE-CIS -> derived FEATURE_COLUMNS
scripts/run_loop.py              # run the loop, write results/
scripts/generalisation_check.py  # leave-one-vector-out recall on an unseen attack family
scripts/mitigation_report.py     # calibrate, decide, price the policies
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

[`scripts/prepare_ieee.py`](scripts/prepare_ieee.py) derives all twelve features from raw
IEEE-CIS: velocity, recency and amount-to-average are computed **within a reconstructed
account** (`card1 + addr1 + first-seen-day`, the community-standard uid) over the real time
ordering, using only transactions strictly before the row they describe. `account_age_days`
is the dataset's own `D1`; `is_cross_border` is `addr2 != 87`; `merchant_risk` is a smoothed
historical fraud rate fit **out-of-fold on the training split only**. Every approximation is
documented in that file's header.

The split is **temporal** — the first 75% of the window trains, the last 25% tests. A random
split leaks future fraud patterns backwards and inflates every metric. The plausibility
manifold used by the evasion optimizer is computed on **train only**.

### The synthetic fallback, and why it is not the default

`load_base_data(source="synthetic")` keeps the original programmatic distribution so the repo
still runs end to end with no download. It should never be quoted, and measuring it against
real data shows why:

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
