# Chhal — a closed-loop adversarial engine for GenAI payment fraud

**Mastercard Innovation Challenge @ GFF 2026 · AI Defense Lab for Payment Security**

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
   (4 GenAI attack           OPTIMIZER                  (LightGBM + SHAP,
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

**No leakage:** the base train/test split is frozen before any attack is injected; the
benchmark and pressure attacks touch neither.

A representative run (8 iterations): benchmark recall **~0% → ~87%** while **FP on legit
stays under 1%**; the hero vector `threshold_hugging` stays hardest to catch (~0.70) because
it mimics normal behaviour. Numbers vary by seed — reproduce with `scripts/run_loop.py`.

---

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

python scripts/run_loop.py --fast     # ~1 min smoke run (4 iterations)
python scripts/run_loop.py            # full 8-iteration run -> results/

streamlit run dashboard/app.py        # the 3-panel live demo (replays results/)
python -m pytest tests/ -q            # contract + optimizer + loop smoke tests
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

| Vector | Idea |
|---|---|
| `threshold_hugging` **(hero)** | LLM-tuned sequences that sit just under every velocity/amount rule and mimic the victim's normal behaviour — the hardest to catch, the heart of the arms race |
| `bustout` | GenAI synthetic identity ages a clean account, then busts out in a burst |
| `card_testing` | Agentic BIN/card probing sized to stay under velocity limits |
| `upi_collect` | 🇮🇳 fraudulent UPI collect-request + rapid drain (India rail) |

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

- **On-manifold rate ~100%** — even after full evasion optimisation, attack feature values
  stay inside the realistic manifold bounds. Direct proof the plausibility guardrail holds:
  attacks evade the detector *without* becoming physically impossible.
- **Per-vector KS distance from legit traffic** — the hero `threshold_hugging` sits **closest
  to legit** (KS ~0.27), which is exactly why it's the hardest to catch; overt vectors
  (`bustout`, `card_testing`) sit further out **by design** — that separation is the fraud
  signal, not a defect. The ranking lines up with detection recall: stealthier ⇒ harder.

`results/fidelity.png` overlays the mimicry vector on legitimate traffic. Base data is
generated; swap in real PaySim / IEEE-CIS and every number becomes a real-data report.

## Repository layout

```
chhal/
  contract.py      # AttackBatch, ScoreReport, FEATURE_COLUMNS — the frozen interface
  data.py          # base distribution (swap in PaySim / IEEE-CIS here); frozen train/test
  detector.py      # LightGBM + SHAP blue-team detector
  redteam/         # the four live-loop attack vectors
  optimizer.py     # constrained evasion optimizer (the novel core)
  evaluation.py    # held-out split protocol + metrics
  fidelity.py      # KS-tests + on-manifold rate — fidelity as a metric, not a claim
  loop.py          # orchestration -> the arms-race curve
scripts/run_loop.py    # run the loop, write results/
dashboard/app.py       # 3-panel Streamlit demo (replays results/)
tests/                 # contract + optimizer + loop smoke tests
```

## Data

The base distribution is generated programmatically so the repo runs with one command and
is fully reproducible. [`chhal/data.py`](chhal/data.py) is a single swappable
function — point `load_base_data` at real **PaySim** or **IEEE-CIS** features and nothing
downstream changes, because everything downstream only knows `FEATURE_COLUMNS`.

## Team

Demilade · Akshat. See [`STRATEGY.md`](STRATEGY.md) for the full concept, taxonomy, and
scoring rationale.

*Metrics in the deck are goals to hit and measure; the held-out protocol is what lets us
report them credibly.*
