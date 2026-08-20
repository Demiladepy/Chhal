# Chhal — a closed-loop adversarial engine for GenAI payment fraud

**Mastercard Innovation Challenge @ GFF 2026 · AI Defense Lab for Payment Security**
**Team:** Demilade + Akshat (2) · **Submission deadline:** 31 Aug 2026

> *Chhal* (छल) — *deception*. Our system is a self-enclosing loop: every deception it
> invents becomes the training ground for a defense that learns to see through it, and
> every gap the defense reveals feeds the next deception.

*v2 — strategy refined: loop contract specified, evaluation protocol made defensible,
evasion optimizer constrained to plausibility, tabular vs. agentic tracks separated.*

---

## The one idea that wins

Most teams will submit **three disconnected things**: a slide of attack ideas, a data
generator, and a fraud classifier. The brief tells us — twice — what actually wins:

> *"The best solutions turn their own simulated attacks into the training ground for a stronger defense."*

So we don't build three things. We build **one closed loop** that gets stronger every
iteration, and we make the *loop itself* the demo:

```
        ┌─────────────────────────────────────────────────────────┐
        │                                                         │
   [1] RED-TEAM AGENT ──► [2] HIGH-FIDELITY SIMULATOR ──► [3] DETECTOR
   (LLM invents/adapts        (renders realistic              (XGBoost + SHAP,
    attack vectors)            transactions + artifacts)       flags fraud)
        ▲                                                         │
        │                                                         ▼
        └──────────  gaps / missed attacks feed back  ◄───── retrain + score
```

**The money chart** (slide 1 of the deck and in the live UI): detection performance on a
**held-out set of novel attacks**, tracked over loop iterations. See "Making the arms-race
curve defensible" below — the chart only wins if it can survive a judge asking *"isn't this
circular?"* Get that right and one picture proves novelty, efficacy, and the closed loop at once.

**Why we can build this fast:** it's the exact red-team/blue-team + adversarial-evaluation
shape we already work in. We're porting a mental model we know, not learning a new one.

---

## The loop interface contract (LOCK THIS DAY 1 — it is the whole project)

The original plan called this the day-1 task but left it as a TODO. It is not a TODO; it is
the single artifact that lets two people build in parallel without integration hell. Both
sides code against this frozen schema from hour one. Concretely:

**What the red team emits — one `AttackBatch`:**

```
AttackBatch {
  vector_id:      str           # e.g. "threshold_hugging", "bustout", "card_testing"
  iteration:      int           # which loop pass produced it
  transactions:   DataFrame     # SAME columns as the base dataset — no extra fields
  labels:         Series[int]   # 1 = fraud (all rows in an AttackBatch are fraud)
  provenance:     dict          # seed, optimizer params, storyline text (for the write-up/UI)
}
```

**What the detector consumes / returns — one `ScoreReport`:**

```
ScoreReport {
  iteration:      int
  split:          str           # "train" | "heldout_known" | "heldout_novel"
  precision, recall, f1, auc:  float
  fp_rate_on_legit:            float
  per_vector_recall:           dict[str, float]   # which attacks still slip through
  shap_top_features:           list               # for the "why flagged" panel
}
```

**The one rule that keeps the loop honest:** the red team may only emit rows in the **same
feature space** as the base data. No attack is allowed to invent a new column the detector
can't see — otherwise "detection" is trivial and the result is fake. This single constraint
is what forces the arms race to happen in a realistic space.

Lock these two structs and the field lists on day 1. Everything else can change.

---

## Pillar 1 — IDENTIFY  (breadth = points; catalog ~15, build ~4)

A taxonomy of where **GenAI specifically** changes payment fraud. Full list goes in the
write-up (scores "diversity of attacks"); we implement the starred ones.

**Two tracks — and be explicit about which is which (this was blurred in v1):**

- **★ LIVE-LOOP vectors** emit transaction features and flow through the XGBoost detector.
  These *are* the closed loop and the live demo.
- **◆ SHOWCASE vectors** are text/agent/media attacks (voice clone, prompt injection). They
  do **not** emit tabular features, so they cannot feed the tabular detector. We demo them
  qualitatively and score them in the write-up for breadth — we do **not** pretend they close
  the same loop. (Optional bridge: map a showcase attack's *outcome* into a transaction, e.g.
  a successful voice-clone APP scam becomes an anomalous push payment — but only if time allows.)

**Synthetic identity & KYC**
- GenAI identity bundles (face + docs + backstory) passing onboarding — ◆ showcase
- Real-time deepfake defeating video-KYC / liveness — ◆ showcase
- ★ **Synthetic-identity bust-out** — age a clean account, then max out — **live-loop**

**Authorized Push Payment / social engineering**
- Voice-clone CEO / "hi mum" transfers — ◆ showcase
- AI call-center agent passing knowledge-based auth (account takeover) — ◆ showcase
- ★ **UPI collect-request scam** (India rail — see edge below) — **live-loop** (as an
  anomalous inbound-collect + rapid-drain transaction pattern)

**Adversarial ML evasion**
- ★ **Mimicry / threshold-hugging** — LLM/optimizer-guided transaction sequences that sit
  just under velocity & amount rules and imitate the victim's normal behaviour.
  ***This is the hero live-loop vector and the technical heart of the arms race.*** If we
  cut everything else, this one stays.
- ★ **Intelligent card-testing / BIN probing** that adapts to velocity limits — **live-loop**

**Agentic-commerce fraud (our unfair advantage — but it's a SHOWCASE, not the loop)**
- ◆ **Prompt-injection against AI shopping/payment agents** — redirect a payment or
  exfiltrate card credentials by poisoning the agent's context. Almost nobody will submit
  this; it's literally our agent-safety research domain. It wins novelty points in the
  write-up and a short scripted demo — it does **not** feed the tabular detector, and we
  say so plainly rather than fudging it.

**Post-fraud**
- GenAI friendly-fraud / chargeback narratives · automated mule recruitment — ◆ showcase

### 🇮🇳 The India / UPI edge (strategic)
GFF is in **Mumbai**, judged on **"real-world feasibility in live payments."** UPI is the
dominant rail. Grounding attacks in **UPI collect-requests, fake-merchant QR, and
IMPS/RTP** — not just US card-present fraud — reads as far more feasible than a generic
Kaggle-credit-card solution. Cheap to do, high-signal to these judges.

---

## Pillar 2 — GENERATE  (fidelity is judged → anchor to real data, never invent from scratch)

- **Base distribution:** **PaySim** (agent-based mobile-money simulator — perfect thematic
  fit) and/or **IEEE-CIS Fraud Detection** (rich real features). We match amount, timing,
  merchant-category, and velocity distributions so injected fraud is statistically plausible.
- **Attack layer (LLM agents):** produce the *storyline* (phishing text, voice script,
  synthetic-identity profile) **and** the transaction feature pattern per vector.

### The evasion optimizer — the novel part, now with the plausibility guardrail

The optimizer nudges an attack's features to evade the **current** detector. This is what
makes it a *loop*, not static generation. But the naive version is a trap: optimize features
straight at XGBoost's decision boundary and you get "attacks" that fool the model but that
**no real fraudster could execute** — impossible timing, contradictory fields, off-manifold
values. That would tank the "real-world feasibility" score and a sharp judge would catch it.

So the optimizer is a **constrained** search:

- **Objective:** minimize detector fraud-score (maximize evasion).
- **Method:** gradient-free (random / evolutionary / simple hill-climb) perturbation of a
  *bounded* set of attacker-controllable features. No gradients through XGBoost needed —
  keeps it simple and honest.
- **Hard constraints (the guardrail):** every candidate must (a) obey business rules
  (velocity caps, amount ranges, valid merchant categories), (b) stay inside the realistic
  manifold (within KS/quantile bounds of real fraud+legit), and (c) only touch features an
  attacker actually controls (amount, timing, sequence — **not** e.g. issuer-side risk scores).
- **Result:** attacks that evade *and* remain executable. That distinction is a talking point,
  not a footnote — it's what separates us from a team that just adversarially perturbs a CSV.

- **Prove fidelity in the deck:** overlay real vs synthetic feature distributions
  (KS-test / histograms). A fidelity *metric* beats a fidelity *claim*. (Per-feature KS is a
  floor, not a ceiling — mention we also eyeball joint/velocity structure so we're not
  overclaiming.)

---

## Pillar 3 — DEFEND  (accuracy up, false positives down)

- **Detector:** **XGBoost / LightGBM** — pragmatic SOTA for tabular fraud: fast, strong,
  interpretable, deployable (ticks "feasibility"). **SHAP** for "why this was flagged."
- **Report:** precision, recall, **F1, AUC**, and **FP rate on legitimate payments** —
  and crucially the **delta after each loop iteration**, on the held-out splits below.
- *Stretch (only if time):* a small **graph** layer (accounts/devices/beneficiaries) for
  mule rings, and an LLM classifier for social-engineering text. Cut first if behind.

### Making the arms-race curve defensible (the part that turns a pretty chart into a winning one)

The whole result hinges on one question a judge *will* ask: **"Isn't this circular — the red
team optimizes against your detector, you retrain on those, of course it improves?"** If we
can't answer that, the money chart is worthless. The fix is a clean evaluation protocol:

- **Split attacks, not just rows.** Each iteration the red team produces attacks; we split
  them into `train` (detector may retrain on these) and **`heldout_novel`** (detector *never*
  trains on these — they test generalization to attacks it hasn't seen).
- **The curve we plot is `heldout_novel` F1/AUC over iterations**, not train performance.
  Rising held-out performance means the detector is learning the *shape* of adaptive fraud,
  not memorizing specific attacks. That is the honest, defensible claim.
- **Expect — and show — oscillation, not a clean monotonic line.** A real arms race dips when
  the red team finds a new evasion, then recovers when blue retrains. A suspiciously smooth
  upward line reads as fake. The *trend* rising through the oscillation is the story;
  annotate the dips ("red team discovered X → recovered in N iterations").
- **No leakage:** base-data train/test split is frozen before any attack is injected; the
  `heldout_novel` attacks touch neither. State this explicitly in the write-up — ML-literate
  judges look for exactly this.

---

## The web prototype (3-panel dashboard = the live demo)

| Left — Red Team | Center — Live Stream | Right — Blue Team |
|---|---|---|
| pick/spawn attack vectors; watch attacks generate | transactions flowing, fraud flagged in real time | metrics + **the held-out arms-race curve over iterations** |

**Stack:** **Streamlit** (presentable UI in ~a day) or FastAPI + light React. Streamlit for a
2-person, 2-week build. Pre-compute the loop offline and let the UI *replay* it — do not try
to run live training inside the demo (it will stall on stage). The dashboard reads results;
it doesn't compute them live.

---

## How it scores on every judged criterion

| Criterion | How we win it |
|---|---|
| Diversity of attacks | ~15-vector GenAI taxonomy, live-loop vs showcase clearly split, India rails included |
| Fidelity of simulation | PaySim/IEEE-CIS base + distribution-match evidence (KS-test) + plausibility-constrained attacks |
| Detection efficacy | XGBoost + SHAP; F1/AUC/FP reported per iteration on **held-out novel attacks** |
| Novelty | the **closed adaptive loop** + constrained evasion optimizer + agentic red team + UPI grounding |
| Real-world feasibility | deployable detector, attacks constrained to executable manifold, live UPI/RTP framing, streaming demo |
| Defensibility (implicit) | held-out protocol + no-leakage statement + honest oscillating curve — survives judge scrutiny |

---

## Anti-patterns that would sink us (pre-empt these)

- **Circular curve** — red team optimizes against the detector, blue trains and scores on the
  same attacks. Fix: the `heldout_novel` split above. This is the #1 credibility risk.
- **Off-manifold "attacks"** — features that fool XGBoost but are physically impossible.
  Fix: the plausibility guardrail on the optimizer.
- **Fake unification** — pretending prompt-injection/voice-clone flow through the tabular
  loop. Fix: the live-loop vs showcase split; say plainly what closes the loop and what doesn't.
- **Live training on stage** — demo stalls. Fix: pre-compute, UI replays.
- **Breadth over depth in code** — 15 half-built vectors beat by 4 that actually loop. Fix:
  breadth in the write-up, depth in exactly 4 live-loop vectors.

---

## Two-person split (flexible)

- **Demilade — Red side + loop:** IDENTIFY taxonomy, the LLM red-team agent, attack
  injection, the constrained evasion optimizer, and the loop orchestration. (Closest to our
  red-team research.)
- **Akshat — Blue side + product:** detector + features + metrics, the base-data pipeline,
  the **held-out evaluation protocol**, the Streamlit dashboard, repo hygiene/reproducibility,
  and the deck. (Plays to the stronger-engineering role.)
- **Shared:** the loop interface contract above (`AttackBatch` ↔ `ScoreReport`) — lock it on
  day 1 so both sides build in parallel.

---

## Realistic 2-week plan

- **Days 1–2:** lock the `AttackBatch`/`ScoreReport` contract + data schema + the train/test/
  heldout split policy; stand up the base dataset; XGBoost baseline on unmodified data
  (get a real F1/AUC number).
- **Days 3–6:** red-team agent + the hero vector (threshold-hugging) + 2 more injected; first
  end-to-end loop pass with the held-out split wired in; Streamlit skeleton.
- **Days 7–9:** the constrained evasion optimizer + retraining loop; the held-out arms-race
  curve; SHAP; UPI vector.
- **Days 10–12:** fidelity plots, deck, polish the UI (replay mode), write the taxonomy doc,
  README + reproducibility, **submit before 31 Aug** (draft/un-submitted work is not judged).

**Scope discipline:** breadth in the *write-up*, depth in *4 live-loop vectors of code*. Cut
the graph layer, LLM-text detector, and showcase-to-tabular bridge first if behind. The loop +
defensible held-out arms-race curve + dashboard is the non-negotiable core.

---

## Honest note

This runs in the same two weeks as the fellowship paper, the 108-trace labelling, and exams.
It's winnable for two people **only if scoped this tightly** — the closed loop is the whole
game, everything else is optional polish. If a week gets crushed, protect the loop demo and
drop breadth, not the other way around.

*(Targets in this doc — fidelity match, detection F1/AUC — are goals to hit and measure, not
results we already have. The held-out protocol is what lets us report them credibly.)*
