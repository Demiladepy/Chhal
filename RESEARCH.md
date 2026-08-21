# Competitive Research — Prior Hackathons, Judging Patterns & News Sources

Research pass done 20 Aug 2026, 11 days before the 31 Aug submission deadline. Covers: what similar
competitions have run before, what made winning submissions win, how red-team/blue-team AI security
contests typically judge diversity/fidelity/efficacy, and where to track ongoing GenAI-fraud news.

---

## TL;DR — actionable synthesis

**No prior edition of this exact event exists** — the Mastercard Innovation Challenge @ GFF 2026 is a
debut. Nobody has run a live competition with Chhal's actual closed loop (attack model regenerates as
the defense adapts, defense retrains, repeat) — that loop itself is genuine white space and the
strongest novelty argument available. The closest precedents below are still worth mining hard.

### What won, and what it means for Chhal

- **IEEE-CIS Fraud Detection (Kaggle 2019, the gold-standard tabular fraud benchmark)** — winners' single
  biggest lever was **entity resolution** (building a synthetic client UID from `card1+addr1+D1`, then
  aggregating per-entity features), not model choice; ensemble of boosted trees beat everything else;
  **time-ordered validation was non-negotiable**. *Check: does the LightGBM detector have any
  per-entity/identity-linking features, or is it scoring rows independently? If independent, this is
  the #1 fixable gap. Also confirm validation is on time-ordered splits, not random.*
- **Razorpay Bumblebee (production, not a hackathon, but the closest real system to this brief)** —
  routed multi-agent architecture (specialist sub-agents per risk dimension) beat a single generalist
  LLM classifier; explicit principle: "prune early, never pass raw unstructured data to LLMs."
- **IEEE SaTML 2024 LLM CTF + MLSEC (closest true closed-loop attack↔defend precedents)** — a defense is
  only scored *after* it clears a **utility/false-positive gate**; attack scoring in MLSEC rewards
  query-efficiency (realistic effort) over brute force. *Report detector efficacy as catch-rate at a
  bounded FPR — a tradeoff curve, not one headline accuracy number.*
- **NeurIPS Trojan Detection Challenge 2023** — organizers rewrote scoring mid-competition because raw
  attack-diversity was gameable by near-duplicate junk; final formula = diversity **gated by** success
  rate. *4 attack vectors that each demonstrably land beats padding with variants of the same vector.*
- **NIPS 2017 Adversarial Competition (Google Brain)** — every submitted attack was run against every
  submitted defense (cross-product), which structurally forces generalization and exposes defenses
  overfit to their own attacker. **The sharpest warning across all research passes**: a closed loop that
  only ever trains/tests against attacks it generated itself is a known credibility red flag. Hold out at
  least one attack pattern the detector never saw during the loop, and show it still catches it.
- **DARPA AIxCC** — patching (defense) explicitly weighted 3× over mere detection in the public scoring
  guide. *Signal: if Mastercard's rubric is vague on attack vs. defense weighting, assume "did something
  useful with it" outweighs "found it."*
- **RBI HaRBInger (HAWK) + Bunq ML Fraud Detector (Devpost)** — both winners paired the model with a
  **usable interface** (OCR+API integration; analyst review UI), not just a notebook/repo. *The Streamlit
  dashboard should function as an actual human-in-the-loop review tool (flag → investigate → resolve),
  matching Mastercard's explicit ask for a "working web prototype."*
- **"Adversarial fraud sample generation with reinforcement learning" (2026, ScienceDirect)** — the
  closest published architecture to Chhal's exact idea: RL generator + closed-loop retraining on
  synthetic fraud folded back into the classifier. Cite this directly in the deck as academic grounding —
  no live competition has run this loop before, so this paper + MLSEC/SaTML/AIxCC for judging-rubric
  precedent is the strongest "we know this space" signal available to judges.

### Concrete gaps to address before Aug 31, ranked by impact/effort

1. **Biggest strategic risk: the 4 attack vectors are tabular, but the brief explicitly names deepfake
   KYC, fake merchant storefronts, synthetic identities.** FREUID, DFDC, and the vishing precedents
   (Battle of the Bots, John Henry) all show judges in this exact space expect the "Gen" in GenAI fraud to
   mean an actual generated artifact (text/image/document), not just perturbed transaction rows. If time
   allows, even one vector that generates a synthetic identity/document/merchant-listing snippet (not
   just numbers) will read as far more on-brief than four purely tabular vectors. If time doesn't allow a
   new vector, at minimum reframe existing vectors' narrative around a GenAI generation step.
2. **Add entity-linking features to the LightGBM detector** — cheapest, highest-leverage fix from
   IEEE-CIS's playbook. Even a rough per-entity aggregation (txn count/mean/time-since-last for a
   synthetic ID) will visibly upgrade the detector's credibility.
3. **Switch/confirm time-ordered validation**, not random split — a five-minute check that prevents an
   obviously inflated demo number.
4. **Hold out at least one attack pattern from the closed-loop training** and show the detector still
   catches it — directly answers the "is this just self-play" question a sharp judge will ask.
5. **Report detection efficacy as a catch-rate-vs-false-positive curve**, not a single accuracy figure —
   matches SaTML/MLSEC/AIxCC judging patterns and reads as rigor.
6. **Make the Streamlit dashboard a review workflow** (flag → why flagged → analyst action), not a
   metrics wall — matches the Bunq/HAWK pattern of what won.
7. **Cite prior art explicitly in the deck**: PaySim/AMLworld (synthetic-data legitimacy), the 2026 RL
   adversarial-fraud paper (closest architecture ancestor), MLSEC/SaTML/AIxCC (judging precedent). Costs
   nothing but writing and signals "we did the homework" — no other team is likely to have this.
8. **Scope calibration**: no prior edition of this exact event exists; closest comparators are RBI
   HaRBInger and IEEE-CIS, not DEF CON-scale red-teaming. Don't over-build breadth — a tight,
   well-instrumented loop beats a sprawling one with 11 days left.

### Newsletter — pick one combo

**Primary: PYMNTS (Fraud/Scams vertical)** — <https://www.pymnts.com/trends/scams/> — daily, free, the
only source found mapped directly onto "GenAI payment fraud" (deepfake scams, synthetic identity,
agentic fraud), which is literally this competition's topic.

**Alternate: tl;dr sec** — <https://tldrsec.com/subscribe> — weekly, free, best signal-to-noise on
prompt injection/agentic-AI attacks/AI red-teaming tooling, which is the technical backbone of the
red-team half.

---

## 1. The target competition — confirmed primary-source details

**"Mastercard Innovation Challenge @ GFF 2026"** (official name; press bio-lines calling it "The AI
Defence Lab for Payment Security" appear to be the tagline/theme, not a separate event).

- **Host:** Mastercard AI Garage
- **Registration/info page:** <https://luma.com/kyz978xv>
- **Context:** runs alongside Global Fintech Fest (GFF) 2026, Mumbai, Sept 8–11 — a Mastercard-run
  track, not an official GFF hackathon (absent from the official GFF hackathon listing at
  <https://www.globalfintechfest.com/gff-hackathons>, which only lists PSB Hackathon Series, SBI
  Hackathon, SEBI Securities Market TechSprint, and NABARD Hackathon)
- **Format:** red team/blue team, end-to-end attack-and-defense adversarial AI system
- **Dates:** registration Aug 10–20, submission deadline Aug 31, results Sept 5, in-person finale
  Sept 8–11 at GFF 2026
- **Prize:** ₹2,56,000 / ₹1,28,000 / ₹64,000 (1st/2nd/3rd) + showcase slot at Mastercard's GFF booth

**No prior edition of this exact competition exists** — 2026 appears to be its debut. Nothing found
under "Mastercard Innovation Challenge" or "AI Defence Lab" at GFF 2023, 2024, or 2025. Other
Mastercard-run hackathons (useful adjacent signal, none an exact precedent):

- Mastercard x FAB "AI Startup Challenge" (UAE, Mar 2025) — won by Teammates.ai, $150K
- Mastercard problem statements inside MAS Singapore's Global Fintech Hackcelerator 2025 ("AI for
  Financial Services" track, alongside BNP Paribas & Prudential)
- Mastercard x BrainStation campus hackathon — winning idea "Cloak," a disposable credit-card-number
  generator for CNP fraud/ID-theft reduction
- Mastercard Cybersecurity Hackathon, Ulster University
- Mastercard + AUC Data Science Initiative financial-inclusion hackathon
- Mastercard/CapitalOne/Visa co-sponsored hackathons via AngelHack at Money20/20
- ("Mastercard Interns Global Innovation Challenge" is an internal intern program, unrelated to the GFF
  public hackathon.)

---

## 2. Foundational reference competitions/datasets

### IEEE-CIS Fraud Detection (Kaggle, 2019) — the gold-standard reference

Hosted by IEEE Computational Intelligence Society + Vesta Corporation. Ran Jul–Oct 2019; 6,381 teams,
7.4K+ competitors, 104 countries. <https://www.kaggle.com/competitions/ieee-fraud-detection>

What made winning solutions win:

- **Entity resolution was the single biggest lever, not model choice.** The "magic feature" (credited to
  Kaggle GM Chris Deotte) was a synthetic client/UID from `card1 + addr1 + D1` (D1 = days since card
  first used), de-anonymizing "same physical cardholder" across rows. Per-client aggregations built on
  top of that UID (~44–47 features) dominated feature importance.
- **Ensemble of gradient-boosted trees**, not deep learning: CatBoost + LightGBM + XGBoost, blended or
  stacked. CatBoost alone hit 0.9639/0.9408 (public/private AUC); the ensemble reached ~0.9677/0.9459.
- **Time-respecting validation was non-negotiable** — GroupKFold on ordered months, never letting future
  data leak into training.
- **Post-processing trick:** once a transaction's UID/client is known, replace all of that client's
  individual predictions with the client's average prediction.
- **Feature hygiene**: fit a single-feature model for every engineered feature and drop anything scoring
  below 0.5 AUC before it goes near the final model.
- Frequency/target encoding on categoricals; splitting amount into dollars/cents (fraud clusters at
  "clean" cent values); normalizing drifting columns into stable per-client constants.

Reference repos: <https://github.com/arunm8489/IEEE-CIS-Fraud-detection> ·
<https://github.com/Aziko13/IEEE-CIS-Fraud-Detection> ·
<https://github.com/KovalevEvgeny/kaggle-fraud-detection>

**Takeaway:** whatever GenAI/red-team layer gets built, the blue-team detector should still lean on this
playbook — entity/identity resolution + boosted trees + strict time-based validation is still the state
of the art for tabular payment fraud, and judges will very likely have this exact mental model.

### PaySim (synthetic mobile-money dataset)

Simulator built from a real African mobile-money provider's aggregated statistics, fraud injected
synthetically for public release — directly analogous to generating a GenAI attack corpus responsibly.
~6.3M transactions, 5 txn types, 0.13% fraud rate (realistic class imbalance).
<https://www.kaggle.com/datasets/ealaxi/paysim1> · paper: <https://www.researchgate.net/publication/313138956>

Worth citing PaySim's methodology (agent-based simulation calibrated against real aggregate statistics)
as prior art for the synthetic-attack generator, rather than reinventing it from scratch.

### ULB/Worldline Credit Card Fraud dataset

2013 European cardholder transactions, 284,807 txns, 492 frauds (0.172%), PCA-anonymized features.
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> — canonical "imbalanced classification 101"
benchmark (SMOTE, class-weighting, precision-recall-AUC over ROC-AUC).

---

## 3. Other payment-fraud hackathons (platform-hosted)

- **RBI HaRBInger** (Reserve Bank of India's global hackathon — most authoritative Indian-regulator
  fraud/banking hackathon, directly relevant since this event is also India-based/GFF-adjacent):
  - 2021 ("Smarter Digital Payments"): runner-up Ezetap/Razorpay's **HAWK** fused social-media scraping
    for fraud-complaint keywords + sentiment analysis + TrueCaller verification + OCR to pull phone
    numbers/VPAs out of screenshots. Judges rewarded unconventional data sources fused with practical,
    deployable integrations — not just raw model accuracy.
  - 2025 (4th edition), "Secure Banking: Powered by Identity, Integrity and Inclusivity": 496 teams/15
    countries, 21 finalists. Winners included an AI-driven fraud detection/compliance platform and a
    Tokenised-KYC system. ₹40L winner / ₹20L runner-up per problem statement.
- **HackerEarth — SuRaksha Cyber Hackathon** (Canara Bank): behavior-based authentication for mobile
  banking, real-time anomaly detection in behavioral patterns.
- **Devpost fraud/payments hackathons:** FSI Hackathon (identity+payment fraud prevention); FinanceAI
  Telecom Paris Hackathon (explicitly scored on generalizing to clients never seen in training — a good
  judging-rubric idea to borrow); DeveloperWeek 2025 (fraud track backed by Fingerprint); Bunq ML Fraud
  Detector (REST-API ML backend + analyst review UI — judges rewarded a usable human-in-the-loop
  interface, not just a notebook).
- **Visa hackathons:** Visa Disrupt SF 2018 and Visa/Money20/20 2018 ran fraud/payments-experience
  tracks; a TechCrunch Disrupt Visa Challenge winner combined ML fraud scoring with haptic-feedback UX.
- **MAS Global Fintech Hackcelerator (Singapore)** — recurring annual hackcelerator that has repeatedly
  included Mastercard-authored problem statements; 2025 theme was "AI for Financial Services."

---

## 4. GenAI-specific attack/defense prior art

**Bottom line:** no competition found does exactly what Chhal proposes — a red-team GenAI generating
novel payment fraud attacks at scale, closed-loop into a blue-team detector that retrains on them.
Real white space. Closest prior art splits into three buckets.

### 4.1 Directly on-topic: GenAI generating the attack, humans/models detecting or executing it

- **Battle of the Bots: Vishing Edition** — DEF CON Social Engineering Community (SEC Village), DEF CON
  33 (2025) / 34 (2026). Teams build autonomous LLM+voice-synthesis agents that place live vishing calls
  against real human targets, scored on objectives captured. ~1,500-word prompts sufficed, no jailbreak
  needed — direct blueprint for a "red-team agent generates the attack content" module.
- **John Henry Competition** — DEF CON SEC Village CTF. AI-driven vishing bots vs. human social
  engineers, head-to-head live calls, 22-minute window. AI scored 17 objectives vs. 12 for the human
  team — demonstrates AI-generated fraud/social-engineering attacks outperforming humans in a scored
  contest. Academic writeups: arXiv:2409.13793, arXiv:2607.09970.
- **The FREUID Challenge 2026** (IJCAI-ECAI 2026, Bremen; host: Microblink Fraud Lab) — Kaggle-hosted
  benchmark explicitly scoped to detect identity document fraud including **GenAI-driven digital edits**.
  Closest thing found to a formal challenge where the fraud corpus is explicitly GenAI-generated.
  <https://freuid2026.microblink.com/> · <https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai>
- **ASVspoof** (biennial, INTERSPEECH-affiliated, since 2015) — organizers generate spoofed/synthetic
  speech via TTS/voice conversion, participants build countermeasures. Most mature/longest-running
  voice-fraud generation-vs-detection benchmark.
- **Deepfake Detection Challenge (DFDC), 2019–2020** — Facebook/Meta, AWS, Microsoft, Partnership on AI.
  $1M prize pool, 128k+ face-swap/deepfake clips generated, ~2,300 Kaggle teams. Largest precedent for
  "generate the synthetic attack corpus at scale, now detect it" — useful scale reference.

### 4.2 Red-team vs. blue-team / attack-then-defend competitions (general methodology)

- **MLSEC** (Microsoft, NVIDIA, CUJO AI, VM-Ray, MRG Effitas; annual since ~2019) — Defender track
  (submit a classifier) vs. Attacker track (evade other teams' classifiers) on real malware/phishing
  corpora. Best precedent for "generate a fraud artifact a real blue-team model has to catch," scored
  adversarially.
- **NeurIPS Trojan Detection Challenge (TDC), 2022/2023** — Center for AI Safety. Paired Red Teaming +
  Trojan Detection tracks, $30k pool. Cleanest "red-team attacks, blue-team detects, both scored
  together" template outside of fraud.
- **CSAW HackML** (NYU, annual since ~2019) — red-team-inserted neural backdoors, blue-team detection.
- **DEF CON AI Village Generative Red Team (GRT) Challenge, DEF CON 31 (2023)** — 2,244 hackers
  red-teamed 8 LLMs over 2.5 days across 21 harm topics, 17k+ scored conversations — largest public
  LLM red-team dataset to date.

### 4.3 Academic prior art directly matching Chhal's mechanism (no live competition, closest published version)

- **"Adversarial Fraud Generation for Improved Detection"** — Pandey et al., ACM ICAIF '22. GAN generator
  simulates fraud conditioned on genuine transactions, targeting the class-boundary region GANs
  normally under-represent. DOI 10.1145/3533271.3561723.
- **"FraudDiffuse: Diffusion-aided Synthetic Fraud Augmentation"** — ACM ICAIF '24. Diffusion model
  instead of GAN for fraud-sample synthesis. DOI 10.1145/3677052.3698658.
- **"Adversarial fraud sample generation with reinforcement learning"** (2026, ScienceDirect) — RL-based
  "Counterfeiter" generator, multi-objective reward (fidelity + domain consistency), explicit
  **closed-loop feedback**: synthetic samples dynamically evaluated and folded back into classifier
  training. **Architecturally the closest published thing to Chhal — read in full.**
  S156849462600640X.
- **"Adversarial Learning in Real-World Fraud Detection: Challenges and Perspectives"** — ACM Data
  Economy Workshop '23 / arXiv:2307.01390. Survey-level framing of the red-team-generator/blue-team-
  detector arms race in production fraud systems — good for a pitch deck's "why this matters" section.
- **IBM AMLSim / AMLworld** (IBM Research + ETH Zurich, NeurIPS 2023 Datasets & Benchmarks) — multi-agent
  simulator generating synthetic transaction graphs with embedded money-laundering patterns, standard
  reference dataset-generation methodology judges will likely compare against.

### 4.4 Fraud hackathons found that are NOT GenAI-generation-focused (context only)

- **National Fraud Prevention Challenge (NFPC)** — RBIH x TRYST IIT Delhi, concluded Apr 2026, built
  around MuleHunter.ai mule-account detection. Pure detection, no adversarial-generation component.
- **Mastercard Cybersecurity Hackathon** — Ulster University, general student cybersecurity hackathon,
  no fraud-generation or red/blue framing found.
- **ISB Hackathon on Cybersecurity & AI Safety 2025–26** — BFSI/fintech deepfake/synthetic-fraud track,
  detection-only from what's public.

### Strategic read

Real-world competitions either (a) let humans/AI generate *live* attacks against real targets (Battle of
the Bots, John Henry — vishing only, not scaled/automatable across payment vectors) or (b) let
organizers pre-generate a static adversarial corpus once and freeze it for a detection leaderboard
(FREUID, DFDC, ASVspoof, AMLworld). **Nobody has run a live competition with the actual closed loop** —
attack model regenerates as the defense adapts, defense retrains, repeat. That loop itself is the
pitch-worthy novelty. Cite the ICAIF "Adversarial fraud sample generation with reinforcement learning"
paper as the closest academic grounding, and MLSEC + Trojan Detection Challenge as the closest
competition-format grounding for eval design.

---

## 5. AI red-teaming & adversarial-ML judging methodology survey

How similar competitions actually score diversity / fidelity / detection-efficacy / novelty —
directly transferable to Mastercard's rubric.

- **DEF CON AI Village GRT (2023)** — no automated diversity metric; a panel of independent human judges
  graded submissions (categories like bias needed cultural context). Diversity engineered into the
  category taxonomy design rather than scored as a separate axis.
- **NeurIPS Trojan Detection Challenge 2023** — the clearest lesson of the whole survey: organizers
  **rewrote the scoring formula mid-competition** because raw diversity was gameable — *"the Combined
  Score metric now weights Diversity by Success Rate so that random inputs receive low scores."*
  Diversity must be gated/multiplied by effectiveness, never scored additively. Final ranking used a
  held-out **manual evaluation phase** because automated judges are gameable near decision boundaries.
- **HackAPrompt (1.0/2.0)** — 1.0 used a 3-judge AI panel on intent coverage / level of detail /
  accessibility, ties broken by token count (a feasibility proxy). 2.0 ran **parallel leaderboards per
  criterion** (breadth vs. efficiency) rather than merging diversity and elegance into one score.
- **Gray Swan AI Arena** — blended AI+human pipeline scoring image reliance / harmfulness / originality
  as three explicit named axes; pure automated judges were "finicky," attackers had to multi-turn
  escalate — a warning that a single automated threshold under-rewards realistic single-shot attacks.
- **IEEE SaTML 2024 LLM CTF** (closest true closed-loop attack↔defense competition) — explicit two-gate
  structure: a defense is only eligible for scoring if it maintains **utility** (near-baseline error on
  benchmark tasks) — a defense that just refuses everything is disqualified before ever being attacked.
  Maps directly onto "detection efficacy" needing a false-positive/usability constraint, not raw
  block-rate. Attack scoring measured both breadth of attacker success (coverage) and rate.
- **MLSEC** — attack scoring = evasion combined with **query efficiency** (fewer queries ranks higher,
  rewarding realistic/efficient attacks over brute force); defense scoring gated by bounded
  false-positive rates on real traffic before entering the arena.
- **NIPS 2017 Adversarial Competition (Google Brain)** — the founding closed-loop template: every
  submitted attack run against every submitted defense (full cross-product). Attack score = accuracy
  reduction averaged across every defense (narrow attacks get diluted, implicitly forcing generality);
  defense score = accuracy maintained across every submitted attack (overfit defenses score low). No
  human judge needed — the cross-evaluation matrix itself operationalizes diversity as breadth of
  coverage, not a subjective score.
- **NeurIPS 2018 Adversarial Vision Challenge** — defense/robustness scored specifically against the
  **top-5 attack submissions** from the attack track (not one canned attack) — robustness only credited
  if it holds against multiple distinct attack algorithms.
- **CSAW HackML** — evaluated on a held-out set of backdoors with *different trigger properties than
  training*, i.e. generalization enforced structurally rather than scored as a separate novelty number.
- **DARPA AIxCC** — published a weighted scoring guide: **patching (defense) weighted 3× vs. mere
  identification (attack)**. Clearest public precedent for how organizers weight "found the attack" vs.
  "did something useful about it."
- **NIST ARIA Red-Teaming Tier** — trained human Assessors annotate every interaction against a
  scenario-specific Test Packet hard-coding prohibited vs. permitted outcomes (a fidelity/realism
  control: a probe only counts if it produces a policy-defined bad outcome in real context). Diversity
  enforced structurally via pre-defined risk-category coverage, not scored abstractly.
- **OpenAI, "Diverse and Effective Red Teaming..." (2024)** — the single most useful methodological
  reference. Key findings: pure RL-against-reward training collapses to a narrow set of repeated
  near-identical attacks; gradient/suffix attacks are highly effective but *unrealistic* ("unlikely to
  be requests from real users"); their fix factorizes into (1) generate diverse goals, (2) generate
  effective attacks per goal, rewarding diversity on tactics/style, gated by success.

### What this means for judging strategy on this challenge

1. **Diversity is almost never scored as a raw count — it's gated by effectiveness.** Lead with a small
   number of category-diverse attacks that each demonstrably land, not a long tail of near-duplicates.
2. **Fidelity/realism is judged as a distinct gate against a defined threat model, not a vibe.** Frame
   each attack's fidelity in terms of a plausible real fraud-attacker's actual tooling/access/cost.
3. **Detection/defense efficacy is never scored on block-rate alone — pair it with a utility/false-
   positive constraint.** Show the catch-rate-vs-FPR tradeoff curve, not a single headline number.
4. **Closed-loop competitions consistently reward breadth of coverage over depth on one exploit.**
   Demonstrate the detector against attack types it wasn't specifically designed around — self-play
   against only your own attacks is a known credibility red flag with judges familiar with this space.
5. **Human judgment stays load-bearing even where automation exists**, especially for subjective axes
   like novelty/realism. Write the submission assuming a human will read it, not just a rubric-matching
   script.

---

## 6. Newsletter / ongoing news sources

No single newsletter nails both "GenAI/payment fraud trends" and "AI security/red-teaming" — pair one
fraud-side pick with one AI-security-side pick.

**Best for GenAI/payment fraud trends**

1. **PYMNTS** (AI + Fraud/Scams verticals) — daily, free. Strongest trade-press source specifically on
   GenAI-powered payment fraud: deepfake scams, agentic fraud, synthetic identity, real-time payments
   fraud. <https://www.pymnts.com/trends/scams/>
2. **Feedzai Insights** — irregular (~monthly), free. Vendor content but substantive — deepfake fraud,
   GenAI scam mechanics, annual fraud-trend reports. <https://www.feedzai.com/insights/>

(Sardine and Alloy/ACFE have product-update newsletters, not strong recurring editorial trend content —
skip for this purpose.)

**Best for AI security / adversarial ML / red-teaming**

3. **tl;dr sec** (Clint Gibler) — weekly, free, ~90K readers. Broad AppSec/cloud newsletter that has
   become one of the best recurring trackers of AI/LLM security specifically — prompt injection, agentic
   AI attacks, AI red-teaming tools. Best signal-to-noise weekly read. <https://tldrsec.com/subscribe>
4. **AI Security Newsletter** (AISecHub, Tal Eliyahu) — monthly, free. Most purpose-built pure-play AI
   security newsletter: agent security, prompt injection, adversarial ML, AI supply-chain attacks,
   red-team writeups. <https://github.com/TalEliyahu/AI-Security-Newsletter>
5. **Import AI** (Jack Clark, Anthropic co-founder) — weekly, free. Not security-specific — capabilities/
   policy/compute-trends with occasional safety commentary — useful macro context. <https://importai.substack.com>

**Practical recommendation:** **PYMNTS + tl;dr sec** — daily fraud-industry signal plus weekly technical
AI-security signal, both free, both high frequency. Add AISecHub's monthly digest for deeper
red-team/adversarial-ML technical detail.

---

## 7. Novelty Cross-Check — Academic Literature Deep-Dive (21 Aug 2026)

Follow-up pass, 10 days before the 31 Aug deadline, specifically testing whether Chhal's live
closed-loop claim survives contact with the academic literature (not just competition precedent,
covered in section 4 above). 4 independent research angles — GenAI-generated fraud artifacts,
live/online adversarial retraining loops, multi-modal fraud fusion, and agentic-fraud literature —
cross-referenced and synthesized below.

# Chhal Novelty Cross-Check: Synthesis of 4 Research Angles

## 1. TL;DR Verdict

**Partially novel — survives, but on a narrower claim than "nobody's doing this."** No published system combines all three of: (a) LLM-driven **multi-vector** tabular attack generation, (b) a detector that **retrains** (not just gets attacked), and (c) the loop running **live during actual deployment/competition runtime** rather than as an offline training procedure. That specific triple-combination is unclaimed. But two of its three legs are separately, closely covered: **FRAUD-RLA** (Feb 2025) does adaptive RL attacks on a real tabular credit-card detector — it's just missing the retrain half. **ProFraudGuard** (Amazon, KDD 2026 workshop) does closed-loop adversarial fine-tuning with an LLM generator in a real fraud domain — it's just missing the tabular-payments domain and (unverified) the "live during runtime" framing. Chhal's actual novelty rests on the **combination and the runtime-liveness**, not on any single piece being unprecedented. Don't claim "first ever adversarial fraud loop" — claim "first to close the loop live, during operation, across multiple tabular payment-fraud vectors simultaneously." That claim holds.

---

## 2. What's Already Been Done

Ranked by how close each comes to overlapping Chhal's actual claim (live, multi-vector, tabular, closed-loop, LLM-driven generator).

| # | Paper/System | Venue/Year | URL | Mechanism | Overlap verdict |
|---|---|---|---|---|---|
| 1 | **ProFraudGuard** — Singh, Kumar, Nagarajan (Amazon) | KDD 2026 Workshop on AI for Fraud and Abuse | [amazon.science](https://www.amazon.science/publications/profraudguard-proactive-adversarial-fine-tuning-of-fraud-detectors-with-inverse-reinforcement-learning) | LLM generator simulates fraudulent business-registration attempts vs. a risk detector, trained via "Proactive Adversarial Fine-Tuning" (PAFT) with claimed convergence guarantees | **Closest on paper.** Fraud domain, LLM generator, closed-loop, 2026, explicit co-evolution. Differs: single vector (registration/KYC fraud, not payments), and — critically unverified — whether PAFT runs live during deployment or is an offline pre-ship training recipe. No arXiv version exists, only an Amazon Science abstract page; this is the one item worth chasing down before you finalize your novelty language. |
| 2 | **FRAUD-RLA** — Lunghi, Molinghen, Simitsis, Lenaerts, Bontempi | arXiv:2502.02290, Feb 2025 | [arxiv.org/abs/2502.02290](https://arxiv.org/abs/2502.02290) | PPO agent iteratively adapts attack policy over rounds to evade a real credit-card fraud detector | **Closest on domain** (tabular, payments, real credit-card data, adaptive rounds). Verified directly from the PDF: the detector is explicitly frozen/queried-only across all rounds (Algorithm 1) — it never retrains. This is the cleanest "half of Chhal's loop, published" precedent. |
| 3 | Multi-round adversarial graph-based promo fraud detection (author list unverified — paywalled) | Social Network Analysis and Mining (Springer) / OpenReview, 2025-2026 | [springer](https://link.springer.com/article/10.1007/s13278-025-01566-0), [openreview](https://openreview.net/pdf/a77e2c9622033b215c96dedc6320ad223a96e589.pdf) | Detector retrained/re-evaluated across rounds as a rule-based fraud-behavior generator evolves a fraud graph | Same *shape* as Chhal (repeated detector retraining against an adversary) but graph-structured promo/referral fraud, and the generator is heuristic/rule-based, not an optimizer or LLM adapting to the live decision boundary. Confirms the multi-round-retrain pattern is emerging in fraud research generally, one notch weaker on the generator side than Chhal. |
| 4 | **SHERLOCK** — Lu et al. (JD.com + Beijing Jiaotong Univ) | ACM KDD '26, Aug 2026 | [arxiv.org/pdf/2510.08948](https://arxiv.org/pdf/2510.08948) | Transaction/tabular features fused with product/merchant text via RAG; continuously updated knowledge base ("Data Flywheel") from investigator feedback, validated over a live 90-day production window | Closest published "adapts during live operation" precedent for a multimodal fraud system. But it's reactive/human-in-the-loop (investigator outcomes feed back), not adversarial — there's no generative red-team being fought. E-commerce trust/safety, not payment/card fraud. |
| 5 | **EvoMail** — Huang et al. | arXiv:2509.21129, Sep 2025 | [arxiv.org/pdf/2509.21129](https://arxiv.org/pdf/2509.21129) | Genuine live closed loop: one agent generates evasive phishing emails, a defense classifier retrains against them in real time, repeated | The clearest existing proof that the "generate → evade → retrain → repeat, live" *pattern itself* has been built and published — just single-modality (email text), not tabular, not payments. Good evidence the architecture is sound; bad news is it means "live adversarial retrain loop" as a bare pattern is not itself unclaimed — your novelty is domain + multi-vector + tabular, not the loop concept alone. |
| 6 | Multimodal financial-fraud fusion (Nie, Long, Fang, Gao) — *cross-confirmed by two independent research angles* | Journal of Data and Information Science, 2025, 10(4) | [DOI: 10.2478/jdis-2025-0046](https://www.degruyterbrill.com/document/doi/10.2478/jdis-2025-0046/html) | LLM-derived text-summary vectors fused with 19 financial + 11 governance tabular indicators into a GBDT for corporate accounting-fraud prediction | Text+tabular fusion works and is published — but static, defensive-only, corporate 10-K fraud (not payments), no generative attacker, no loop. Kills "text+tabular fusion is unprecedented" as a bare claim; doesn't touch Chhal's actual mechanism. |
| 7 | LLM-GRU-GAN (unverified — 403-walled, possibly non-peer-reviewed) | ResearchGate, ~2026 | [researchgate.net/publication/405492823](https://www.researchgate.net/publication/405492823_LLM-GRU-GAN_A_Multi-Modal_Adversarial_Framework_for_Transactional_Fraud_Detection) | LLM → semantic graph embeddings, GRU temporal modeling, GAN generates synthetic fraud samples | Architecturally closest name-match to "LLM + adversarial generation + tabular fraud" — but the GAN is doing class-imbalance oversampling, not live red-team evasion against a running detector. Worth a closer read if you have time; too unverified to lean on as-is. |
| 8 | MultiAgentFraudBench (Ren et al.) | ICLR 2026 | [arxiv.org/abs/2511.06448](https://arxiv.org/abs/2511.06448) | LLM agents collude on romance/investment/phishing scams against simulated victims; mitigation tested as static intervention | Confirms "agentic fraud" is now a live 2026 academic topic — but text/dialogue only, no tabular data, no retraining loop, mitigation is a one-shot filter test not adapt-retrain. |
| 9 | AISI Network methodology paper (69 co-authors, Singapore/UK/Japan/Australia/etc.) | arXiv:2601.15679, Jan 2026 | [arxiv.org/abs/2601.15679](https://arxiv.org/abs/2601.15679) | Cross-government exercise on *how to evaluate* agentic AI for fraud risk; no finalized benchmark | Not a system at all — but a government-level admission, Jan 2026, that agentic-fraud evaluation methodology is still nascent. Strong "the field itself says this is unsolved" citation. |
| 10 | Document/identity-fraud generation papers (AIForge-Doc, GPT4o-Receipt, "From Forgeries to Foundation Models" survey) | arXiv 2602–2607.xxxx, 2026 | see below | GenAI-generated fake receipts, forms, IDs, and a survey confirming document-fraud research stays siloed from transaction fraud | Zero tabular overlap — confirms document/image-fraud and transaction-fraud research literatures still don't talk to each other. Relevant only if Chhal considers a future document-vector extension. |

**Deduplication note:** the Nie et al. (JDIS 2025) paper and the SHERLOCK paper were independently surfaced by both the `genai-artifact` and `multimodal-fusion` angles — same papers, consolidated above (rows 4 and 6). The "From Forgeries to Foundation Models" survey (arXiv:2607.01442) was also surfaced by both angles.

---

## 3. What's Still Genuinely Open

Specific, technical, unclaimed territory as of this search:

1. **The full triple-combination itself.** No paper closes all three: LLM-driven generator → multiple, distinct tabular attack vectors → detector retrains → loop executes live within an actual deployment/competition runtime (not as an offline training epoch schedule). FRAUD-RLA has the tabular-payments domain + adaptive attacker but a frozen detector. ProFraudGuard has the closed retrain loop + LLM generator but a single non-payments vector and unconfirmed runtime-liveness. Nobody has both.

2. **Multi-vector diversity in one loop.** Every closed-loop or adaptive-attacker system found (ProFraudGuard, FRAUD-RLA, EvoMail, the graph promo-fraud paper) targets **one** attack pattern. A single live loop juggling threshold-hugging evasion, synthetic-identity bustout, card testing, and a UPI-style scam simultaneously — where the detector's retraining on one vector has to hold against the others too — is not addressed anywhere found.

3. **"Runtime" as an actual architectural claim, not a training-recipe claim.** Several papers use "live," "dynamic," "adaptive," or "proactive" language (ProFraudGuard, SHERLOCK) but on inspection describe either offline training procedures or human-in-the-loop reactive updates, not an adversary-vs-detector loop literally executing during a bounded live event. This distinction is exactly where Chhal can stake a clean, specific, technically defensible claim — but only if the architecture genuinely does this (verify your own implementation matches this description before claiming it).

4. **(Stretch, not currently in Chhal's scope)** Fusing an LLM-generated document/text artifact's derived features into the same tabular retrain loop. Confirmed open by two independent angles and a July 2026 survey stating document-fraud and transaction-fraud research remain siloed. Not worth chasing before Aug 31 unless you have spare runway — flagged here as an honest "if you had more time" answer, not a claim to make about the current 4-vector, tabular-only build.

**One operational flag, not a novelty issue:** one research pass surfaced that NPCI/RBI reportedly discontinued P2P UPI "collect request" (pull) transactions effective Oct 1, 2025. Worth a quick independent confirmation — if true, your UPI-collect-scam vector may be modeling a since-patched mechanism, which a Mastercard/NPCI-adjacent judge could flag. Consider reframing that vector's narrative or swapping to a still-live UPI fraud pattern (QR/deep-link push scams) if confirmed.

---

## 4. Recommended Citations for the Pitch Deck

Beyond the papers already in your background list:

1. **ProFraudGuard** (Amazon, KDD 2026 workshop) — cite as the closest published system and explicitly differentiate: multi-vector tabular payments vs. their single onboarding-fraud vector; LightGBM + engineered features vs. their LLM-only detector; and flag (honestly, in your own words) that their "proactive" framing does not confirm live-runtime execution the way Chhal's does.

2. **FRAUD-RLA** (arXiv:2502.02290) — your cleanest, most defensible contrast citation. "Prior RL attacks on tabular fraud detectors adapt the attacker only, against a frozen classifier (Algorithm 1). Chhal closes this loop by retraining the detector each round on what it catches." This is verified firsthand from the PDF, safe to state with confidence.

3. **AISI International Network methodology paper** (arXiv:2601.15679) — a 69-co-author, multi-government (Singapore AISI leading the fraud strand) admission, Jan 2026, that agentic-fraud evaluation methodology is still nascent. Use it to argue the field itself says live agentic/adversarial fraud evaluation is unsolved — strengthens the "why does this matter" framing without you having to assert it unsupported.

4. **SHERLOCK** (arXiv:2510.08948, KDD '26) — cite as the closest "adapts during live operation" precedent, then contrast: reactive human-investigator feedback loop vs. Chhal's adversarial generator constantly probing the live decision boundary. Useful for showing judges you know the adjacent state of the art, not just claiming a vacuum.

5. **EvoMail** (arXiv:2509.21129) — cite to show the live generate-evade-retrain loop pattern is validated and works in a security-adjacent domain (email/phishing), then note nobody has run it on multi-vector tabular payment fraud. Good for pre-empting the "hasn't this been done with GANs/RL before" question with a direct, honest answer instead of dodging it.

---

## 5. Bottom Line for the Team

Claim this: **"a live, multi-vector, closed-loop adversarial system for tabular payment fraud, where an LLM-driven attacker and a LightGBM detector both adapt repeatedly during actual competition runtime."** That specific combination is not published anywhere found across four independent search passes. Do NOT claim "first adversarial fraud generation loop" or "first LLM-vs-detector arms race" in general terms — those individual pieces exist separately (FRAUD-RLA for adaptive tabular attacks, ProFraudGuard for closed-loop fraud retraining, EvoMail for the live-loop pattern itself), and a judge who knows the space will call that out. Before the deck is final, get someone to nail down whether ProFraudGuard's PAFT is genuinely live/runtime or an offline training recipe — that single fact determines whether your closest competitor is a near-miss or a real overlap. Everything else checked out clean: nothing found erodes the core claim, it just narrows exactly which part of it is actually new.