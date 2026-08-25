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

Full run, 8 iterations on real IEEE-CIS (`scripts/run_loop.py`, ~66s):

| metric | baseline | after the loop |
|---|---|---|
| recall @ **0.1%** of legit flagged | 0.25% | **83.80%** |
| recall @ 0.5% | 0.40% | 99.55% |
| recall @ 1.0% | 0.40% | 99.75% |
| **PR AUC** | 0.0176 | **0.9275** |
| alert rate (share of all traffic) | 0.10% | 1.259% |

For comparison, the naive 0.5 cutoff on the same run: F1 0.0065 → 0.8937, ROC AUC 0.9997, FP on
legit 0.32%. The ROC number is the one to distrust.

The baseline catching almost nothing is not a broken detector — the benchmark attacks
were optimised specifically to evade it, which is the optimizer doing its job.

The claim that survives scrutiny is not this curve. It is the leave-one-out one, and it
is much weaker: **39.6% on an attack family never seen in any form**, against 77.6% on
families the detector was trained on — a 38-point generalisation gap
(`scripts/generalisation_check.py`). Per family: `upi_collect` 67.6%, `bustout` 62.4%,
`card_testing` 23.6%, and the mimicry vector `threshold_hugging` **4.8%**, which is close
to nothing. A detector trained on three fraud families does not thereby understand the
fourth, and the stealthier the fourth is, the less it understands. That is the honest
state of this system and the most useful thing in this README.

An earlier version of this file reported 88.7% here. That number was an artifact: the
evasion optimizer used to perturb four derived timeline features as independent scalars,
which stamped every vector with the same impossible-velocity signature, so a "never seen"
family was not really unfamiliar — it shared the tell. Fixing the optimizer (see
[Attacks are campaigns, not rows](#attacks-are-campaigns-not-rows)) cost 13 points of
headline recall and 49 points of leave-one-out recall. Both were ours to lose.

Numbers vary by seed.

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
python scripts/run_loop.py            # full 8-iteration run -> results/  (~66s on real data)

python scripts/generalisation_check.py  # leave-one-vector-out: recall on an UNSEEN family
python scripts/mitigation_report.py     # score -> action -> money
python scripts/ensemble_check.py        # does a second detector arm earn its place?
python scripts/latency_check.py         # can it run inside an authorization?
python scripts/feature_ablation.py      # why the feature space looks the way it does

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
- **Guardrail binding rate ~9.6%** — the non-tautological number: the fraction of proposed
  perturbations that actually landed *outside* the manifold before clipping and had to be
  pulled back. This is the real evidence the guardrail does work — attacks push against
  the plausibility envelope, they don't just float freely inside it.
- **Per-vector KS distance from legit traffic** — the hero `threshold_hugging` sits **closest
  to legit** (KS ~0.27), which is exactly why it's the hardest to catch; overt vectors
  (`bustout`, `card_testing`) sit further out **by design** — that separation is the fraud
  signal, not a defect. The ranking lines up with detection recall: stealthier ⇒ harder.

`results/fidelity.png` overlays the mimicry vector on 590,540 real IEEE-CIS transactions.
Every distance on this page is a distance from real legitimate traffic — see
[Data](#data--real-not-invented).

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
training window the detector never sees. That pool is then split in half — isotonic is
**fitted** on one half and its error is **measured** on the other, because an isotonic fit
scored on its own rows returns ECE 0.0000 by construction and that is a tautology, not a
result. Measured honestly: ECE 0.0062 raw → 0.0062 calibrated on the held-back half (the
raw LightGBM score is already close to calibrated here; what calibration buys is the
*scale*, which is what gets multiplied by a dollar amount). `test_miscalibrated_scores_degrade_the_policy`
locks this in: squash the scores monotonically — leaving every recall and AUC number
identical — and the policy measurably loses money.

```bash
python scripts/mitigation_report.py
```

Priced on the frozen future (147,635 real transactions + 1,600 adaptive attacks the
detector never saw, 4.49% fraud):

| policy | cost per 1k txns | net cost reduction | fraud loss avoided |
|---|---|---|---|
| do nothing | $10,057.33 | — | — |
| block at `score >= 0.5` (untuned) | $5,056.58 | 49.7% | 54.2% |
| tuned allow/step-up/block (amount-blind) | $3,017.77 | 70.0% | 84.4% |
| **expected-cost policy** | **$2,889.93** | **71.3%** | **84.6%** |

**Two columns, because they are two different questions.** *Net cost reduction* nets the
friction we impose on legitimate customers against the fraud we stop — the number a CFO
signs off on. *Fraud loss avoided* ignores that friction and answers only how much of the
money that would have walked out we kept. An earlier version of this table reported the
first and labelled it the second.

**And four rows, because three of them would have been a straw man.** Beating `score >= 0.5`
is not evidence that expected-cost decisions work; it is evidence that 0.5 is a bad
threshold, and nobody deploys an untuned one. The honest comparator is the best
*amount-blind* ladder that exists — allow below one threshold, challenge between, block
above — with both thresholds tuned on the same cost model, on a slice held back from the
one it is priced against. `tune_two_thresholds` finds that pair exactly rather than by
grid search (sorting by score turns it into prefix sums), so the comparator is the
strongest one available, not a convenient one.

Against it, **amount-awareness plus the capacity cap are worth 1.3 points of net cost
reduction** — $128 per thousand transactions, a 4.2% relative saving. That is the real
size of the contribution. The other 20 points came from tuning a threshold, which any
fraud team already does. `test_the_real_edge_is_measured_against_the_tuned_ladder_not_the_naive_threshold`
asserts exactly this ordering, so the framing cannot silently drift back.

The policy declines **0.243%** of real customers outright, against 0.747% for the fixed
threshold, while stopping or challenging 78.7% of all fraud (73.5% of real fraud, 95.3% of
adaptive attacks). Note that both decline rates rose once the linkage block was added: the
detector is genuinely more confident on real fraud, so blocking becomes economically
correct more often. That is the policy working, but it is a real cost and the cost model
is where to argue about it.

### Whose fraud is being avoided

Detection is reported per segment; the economics was not, and that hid something. A
quarter of the cost denominator is fraud **we generated**, and the policy is far better at
our own attacks than at the real thing:

| segment | do nothing | expected-cost policy | net cost reduction |
|---|---|---|---|
| real fraud + all legitimate traffic | $918,544 | $421,805 | **54.1%** |
| adaptive attacks only | $582,362 | $9,475 | 98.4% |

The blended 71.3% borrows credit from attacks we wrote ourselves. **54.1% is the number
that would survive contact with a production book**, and it is the one to quote.

### An honest split we are not hiding

Recall at 0.1% false positives on real legitimate traffic, by segment:

| segment | recall |
|---|---|
| unseen adaptive attacks (our red team) | **65.4%** |
| real IEEE-CIS fraud | **18.9%** |

Real fraud was **3.6%** before the linkage block was added — see
[Mounting attacks on real accounts](#mounting-attacks-on-real-accounts). It is now 18.9%,
a 5.2x lift, and adaptive-attack recall paid for it: the detector has strong real-fraud
features now and leans on them. That trade is visible rather than hidden, and it is the
right one for a submission that has to work on real payments.

These are thresholded on the **raw** detector score, not the calibrated one. Calibration
is monotone so it cannot change the ranking — but isotonic collapses long runs of scores
onto a single value, and at a 0.1% budget the threshold lands inside such a plateau, where
a tie-break decides whether hundreds of rows count as caught. Detection is a property of
the ranking; the calibrated probability is only needed for the economics.

Reporting the blended 30.0% would hide both halves. The mitigation layer closes more of
the remaining gap — 73.5% of real fraud gets stopped or challenged, because a cheap OTP
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
| **supervised only** | **0.3885** | 0.1702 |
| max fusion (either arm flags) | 0.3075 ✗ | 0.1261 ✗ |
| stacked (anomaly score as a feature) | 0.3760 ✗ | 0.1582 ✗ |

**The anomaly arm alone scores 0.000 on unseen attacks and 0.006 on real fraud.** Both
fusions now lose: `max` badly, stacking by 1.85 points on unseen families for +0.4 on
real fraud. It also costs 32MB of the 34MB model footprint.

**This verdict changed, and the change is the point.** On the earlier twelve-feature
space, stacking was worth +1.25 points and we shipped it. Once the linkage block arrived
and real-fraud recall went from 3.6% to 18.9%, the supervised arm had enough signal that
an outlier score added nothing but false positives. The module docstring did not keep up
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

The twelve hand-derived features caught **3.1%** of real IEEE-CIS fraud at a 0.1%
false-positive budget. Adding the dataset's anonymised entity-linkage counts (`C1-C14` —
how many addresses, devices, emails and cards associate with this card) takes that to
**19.7%**. `scripts/feature_ablation.py`:

| features | n | recall @ 0.1% FPR | PR AUC |
|---|---|---|---|
| the 12 we hand-derived | 12 | 3.06% | 0.189 |
| + linkage counts **we built ourselves** | 20 | 3.22% | 0.195 |
| **+ the dataset's C1-C14** | 26 | **19.73%** | **0.474** |
| + both | 34 | 20.35% | 0.472 |
| + C1-C14 and D1-D15 | 41 | 19.29% | 0.488 |
| + everything incl. all 339 V-features | 388 | 21.63% | 0.471 |

Three things fall out of that table. The linkage block is the entire story — all 339
V-features add under two points on top of it, and `D1-D15` add nothing. We tried to
rebuild the same signal from what we *do* understand (distinct counterparties, addresses,
emails and card attributes per account over time, plus longer velocity windows) and
recovered **+0.16 points**. And putting ours on top of theirs is a wash: marginally better
at the tightest budget, marginally worse at looser ones and on PR AUC — noise, not signal.
They are subsumed, not merely weaker. Whatever C1-C14 aggregate over lives in devices,
phones, IPs and cross-card relationships this dataset does not expose.

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
- Evaluation pools exclude any account seen in training. 34.8% of test accounts also have
  transactions before the temporal cut; excluding them removes the argument entirely.
  (Belt and braces: hosts are all-legitimate, so recognising one would push an attack
  toward *legit* and make detection harder, not easier.)
- Attack transactions are timestamped strictly **after** the host's last real
  transaction. A campaign continues an account; it cannot reach into its past.
- Inherited values are read from the host's last real transaction — the most recent state
  anyone could legitimately know at the moment of takeover.

### What it cost and what it bought

| | before | after |
|---|---|---|
| real IEEE-CIS fraud, recall @ 0.1% FPR | 3.6% | **18.9%** |
| unseen adaptive attacks | 98.8% | 65.4% |
| mimicry vector, KS distance from legit | 0.361 | **0.174** |

Fidelity improved sharply because sixteen of twenty-six features are now literally real
values. Adaptive-attack recall paid 33 points for it, which is the honest consequence of
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
| `threshold_hugging` (hero vector), before | 69.9% | — |
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
| **full path, single transaction** | **1.24 ms** | 1.27 ms | **1.29 ms** |

**~35× headroom** against a 50ms risk-decision budget at p99. Batch throughput is
**171,344 txns/sec** (5.8 µs each) at a batch of 10,000 — that is the nightly-rescoring
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
scripts/feature_ablation.py      # which features carry the real-fraud signal, and why
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
