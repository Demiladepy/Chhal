"""The six live-loop attack vectors and one negative control — each a campaign shape,
not a row shape.

A vector declares two things and nothing else:

  `temporal`          how the attack unfolds once the account is compromised — how many
                      transactions, how far apart, how the amount moves. What came before
                      is not declared here: it is the host account's real history. The base class
                      lays out the timeline and DERIVES amount, hour, day_of_week, both
                      velocities, the inter-transaction gap and the amount-to-average
                      ratio from it, using the same function applied to the 590,540 real
                      transactions. Those seven columns are therefore internally
                      consistent by construction rather than by inspection.

  `static_features`   the few things a fraudster actually chooses — the payee, the rail,
                      the destination. Account age, merchant risk and the entity-linkage
                      counts are NOT here: they are inherited from the real account the
                      campaign is mounted on, because an attacker cannot set them.

Every number is a QUANTILE of real legitimate traffic, never a raw value: (0.35, 0.75)
on `amount` means "the middle of what real cardholders actually spend". Change the
dataset and the vectors re-scale themselves.

The bands and the campaign shapes together are what separate the vectors:
`threshold_hugging` lives inside the legitimate body and moves at a legitimate pace,
`card_testing` lives in the extreme tails and moves in seconds. That separation is what
the per-vector KS table then measures.

Text/agent "showcase" vectors (voice clone, prompt injection) live in the write-up, not
here, because they do not emit tabular features — see the strategy doc.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .base import AttackVector
from .campaign import TemporalProfile


class ThresholdHugging(AttackVector):
    """Per-victim mimicry: the campaign is sized and paced from the compromised card's
    OWN history, not from the population's. The heart of the arms race; if everything
    else is cut, this stays.

    NOT NOVEL, and not to be described as such. Carminati et al., ACM TOPS 21(3) 2018,
    ran mimicry attacks against the Banksealer detector on real data from a large Italian
    bank, with the attacker's variables being exactly amount and timestamp. What is ours
    is the per-victim quantile calibration below, the fixed-FPR measurement, and the
    ablation in `scripts/coordination_check.py` that prices it (0.9 +- 1.0 points, i.e.
    not yet a finding).

    `mimic_host=True` is the whole vector. Without it, "hides in the crowd" means the
    middle of everyone's distribution, which is not where the decision is made: the
    detector scores `amount_to_avg_ratio` and the gap against THIS card's baseline, and a
    population-median attack on a below-median card is visibly wrong on both. With it,
    the same quantile band is read off the victim's own spend and the same cadence off
    the victim's own gaps, so a card that buys coffee gets a coffee-sized attack at
    coffee-buying intervals, and the ratio lands near 1 because it genuinely is.

    It should therefore be the vector with the lowest KS distance from real traffic and
    the lowest recall, and the per-vector fidelity table is where that is checked rather
    than asserted. Half of that prediction is false: `autopay_mandate` is closer to legit
    on both KS measures despite carrying no mimicry at all, and the ordering claim is
    withdrawn in the README. The low-recall half holds.

    The other thing this docstring used to leave implicit: mimicry needs
    MIN_HISTORY_TO_MIMIC real transactions to read a distribution off, and IEEE-CIS
    accounts are short. On the benchmark host pool it actually engages on 28.8% of
    campaigns (40.5% train-side); the rest fall back to the population bands this vector
    exists to argue against. `HostPool.mimicry_engagement` measures it and
    `HostPool.describe` prints it every run. That is a limit of the data rather than of
    the idea — an attacker holding a two-transaction statement has nothing to profile
    either — but it caps what this vector can demonstrate here, and it is a large part of
    why its ablation lands inside the noise.
    """

    new_payee_rate = 0.30    # the victim's usual payees, with the occasional new one
    vector_id = "threshold_hugging"
    storyline = (
        "The attacker profiles the victim's own spending and cadence from the card's "
        "history, then transacts inside that profile — just under every velocity and "
        "amount threshold the victim would themselves trip — so nothing about the "
        "sequence is anomalous FOR THIS ACCOUNT."
    )
    temporal = TemporalProfile(
        txns_per_entity=(3, 9),
        inter_arrival_s=(3_600.0, 172_800.0),      # fallback for a thin-history victim
        amount_band=(0.35, 0.75),                  # read off the victim, not the crowd
        start_hour_band=(0.25, 0.85),
        mimic_host=True,
    )

    def static_features(self, n, rng):
        p = self.p
        return {
            "is_new_beneficiary": p.bernoulli(self.new_payee_rate, n, rng),
            "is_cross_border": np.zeros(n, int),
            "channel_code": p.categorical("channel_code", n, rng),  # real channel mix
        }


class SyntheticBustout(AttackVector):
    """Age a synthetic-identity account to look clean, then max it out in a burst."""

    new_payee_rate = 0.80    # burst to fresh beneficiaries, on an account with history
    vector_id = "bustout"
    storyline = (
        "A GenAI synthetic identity (face + docs + backstory) passes onboarding, ages "
        "quietly for months, then busts out: a sudden burst of high-value transfers to "
        "fresh beneficiaries."
    )
    temporal = TemporalProfile(
        txns_per_entity=(8, 25),
        inter_arrival_s=(120.0, 3_600.0),          # the burst: minutes apart, over hours
        amount_band=(0.90, 0.995),
        amount_trend=1.6,                          # escalating as it empties the account
    )

    def static_features(self, n, rng):
        p = self.p
        return {
            "is_new_beneficiary": p.bernoulli(self.new_payee_rate, n, rng),
            # elevated vs the 0.7% legit / 2.2% fraud base rate, because cashing out
            # abroad is this vector's point — but not so high it leaves the manifold.
            "is_cross_border": p.bernoulli(0.10, n, rng),
            "channel_code": p.categorical("channel_code", n, rng),
        }


class CardTesting(AttackVector):
    """Intelligent BIN/card-testing that adapts probe size to velocity limits."""

    new_payee_rate = 0.90    # many distinct merchants, but probes do repeat
    vector_id = "card_testing"
    storyline = (
        "An agent probes stolen card ranges with many micro-authorizations, spacing "
        "and sizing them to stay under velocity limits until a live card is found."
    )
    temporal = TemporalProfile(
        txns_per_entity=(20, 60),                  # many probes on one stolen range
        inter_arrival_s=(2.0, 120.0),              # seconds to two minutes apart
        amount_band=(0.005, 0.06),                 # tiny probes
    )

    def static_features(self, n, rng):
        p = self.p
        return {
            "is_new_beneficiary": p.bernoulli(self.new_payee_rate, n, rng),
            "is_cross_border": p.bernoulli(0.08, n, rng),
            "channel_code": np.zeros(n, int),                       # card rail
        }


class UpiCollectScam(AttackVector):
    """India rail: a fraudulent UPI collect-request followed by rapid drain.

    Scoped to MERCHANT collect, deliberately. NPCI circular
    `NPCI/UPI/OC/220/2025-26` (29 July 2025) discontinued Person-to-Person collect
    entirely: no P2P collect transaction may be "initiated, routed, or processed" on
    UPI from 1 October 2025. Modelling P2P collect would mean modelling a rail that no
    longer exists.

    Two facts make merchant collect the right target rather than a fallback. First, it
    is what survived — collect requests from merchants still run, and they are the
    higher-limit variant. Second, the P2P rail could never have carried this attack
    anyway: a circular of 31 October 2019 capped P2P collect at Rs 2,000 per
    transaction with 50 successful transactions a day, while this vector's
    `amount_band` of (0.75, 0.96) is $125-$500 of legitimate TRAIN traffic -- which is what
    `BaseProfile` samples from, `data._profile_from_train` having fitted the quantiles on
    the training split; the same band read off test traffic would be $141-$593 -- 5x to 21x
    over that cap on every single hop. So the P2P framing was wrong on
    amount from the day it was written, independently of the 2025 withdrawal.

    The mechanic the attacker actually uses is impersonation of a verified merchant:
    the victim is walked into approving what looks like a checkout collect request.
    Nothing about the generated rows changes -- this is a scoping and storyline fix,
    so every committed number for `upi_collect` remains valid.
    """

    new_payee_rate = 0.85    # fresh VPAs per hop, though not perfectly fresh
    vector_id = "upi_collect"
    storyline = (
        "A GenAI social-engineering script impersonates a verified merchant's checkout "
        "and walks a victim into approving a merchant collect-request -- the higher-limit "
        "variant that survived NPCI's October 2025 withdrawal of P2P collect. The funds "
        "are then drained through a chain of fresh VPAs within minutes."
    )
    temporal = TemporalProfile(
        txns_per_entity=(3, 7),                    # a short chain of fresh VPAs
        inter_arrival_s=(30.0, 600.0),             # "within minutes"
        amount_band=(0.75, 0.96),
        start_hour_band=(0.30, 0.90),              # the victim has to be awake to approve
        amount_trend=0.7,                          # each hop takes less as funds run out
    )

    def static_features(self, n, rng):
        p = self.p
        return {
            "is_new_beneficiary": p.bernoulli(self.new_payee_rate, n, rng),
            "is_cross_border": np.zeros(n, int),
            "channel_code": np.ones(n, int),                        # upi rail
        }


class MuleFanout(AttackVector):
    """Many compromised accounts, one operator, one window — the vector GenAI actually
    changes, because what it makes cheap is running a hundred of these at once.

    Every other vector here is a single-account story: this card, this victim, this
    burst. That is what fraud looked like when a person had to work each account by
    hand. The thing generative models change is not the cleverness of one attack, it is
    that one operator can run a mule network at a scale that used to need a call centre.
    So this vector is deliberately unremarkable per account — two to five transfers,
    sensible amounts, nothing that trips a per-account rule — and its signature lives
    entirely in the fact that a hundred unrelated accounts did it inside the same window.

    What this vector is really for
    ------------------------------
    The frozen feature space has no counterparty. There is no beneficiary id, no
    destination account, no edge between two rows — so the coordination that DEFINES this
    attack is not observable by the detector at all. It can only see each account's own
    small burst, and whatever weak clustering survives in `hour` and `day_of_week`.

    So this vector is built as a controlled experiment rather than as a fifth variation on
    speed and size. It carries `mimic_host` for the same reason `threshold_hugging` does: each
    account is made to look normal FOR ITSELF, which strips away the per-row tells the
    detector would otherwise catch it on. What is left as a difference from
    `threshold_hugging` is almost entirely the synchronisation.

    That makes its recall a measurement of something. If it lands near
    `threshold_hugging`'s,
    coordination is genuinely invisible here and "a graph layer is future work" stops
    being a line in a limitations section and becomes a number. If it lands well above,
    the clustering in `hour` and `day_of_week` is doing the work, and that is worth
    knowing too — it would mean a crude time-bucket feature buys some of what a graph
    would.

    An earlier draft of this vector set cross-border at 0.35 against a 0.7% legitimate
    base rate. It scored 93.4%, and it was being caught on that one column rather than on
    anything to do with the network — which would have made the experiment worthless while
    looking like a good result.
    """

    new_payee_rate = 0.75    # fresh mule destinations, though operators reuse drops
    vector_id = "mule_fanout"
    storyline = (
        "One operator drives a network of mule accounts opened or bought at scale. Each "
        "account moves a modest, unremarkable amount onward to fresh beneficiaries — but "
        "all of them move within the same few hours, before anyone reconciles across "
        "accounts."
    )
    temporal = TemporalProfile(
        txns_per_entity=(2, 5),                    # forgettable on its own
        inter_arrival_s=(60.0, 900.0),             # fallback for a thin-history mule
        amount_band=(0.55, 0.85),                  # read off the account, not the crowd
        start_hour_band=(0.05, 0.35),              # unused while mimic_host is on
        mimic_host=True,                           # locally normal, so only timing is left
        coordinated_window_s=6 * 3_600.0,          # the whole network fires in one window
    )

    def static_features(self, n, rng):
        p = self.p
        return {
            # the one thing a mule account cannot avoid: the money goes somewhere new
            "is_new_beneficiary": p.bernoulli(self.new_payee_rate, n, rng),
            # kept near the other vectors on purpose. Pushing it up makes the vector easy
            # to catch on a single column and destroys what it is here to measure.
            "is_cross_border": p.bernoulli(0.10, n, rng),
            "channel_code": np.ones(n, int),                        # transfer rail
        }


class AutopayMandate(AttackVector):
    """A fraudulent recurring mandate disguised as a subscription — the vector whose whole
    signature is REGULARITY, and a controlled probe of a blind spot the others cannot reach.

    Every other vector here moves fast or moves in a burst: `card_testing` probes in
    seconds, `bustout` empties in an afternoon, `upi_collect` drains in minutes. A detector
    tuned on real fraud learns that shape, and `velocity_1h`, `velocity_24h` and the
    inter-transaction gap carry a large part of it. This vector deliberately does the
    opposite. A GenAI phishing flow — disguised as a KYC re-verification or a delivery
    reschedule — tricks the victim into approving a recurring auto-debit *mandate* rather
    than a one-off payment: the live UPI-AutoPay successor to the P2P collect-request that
    NPCI closed in October 2025. The fraud then draws a modest, near-constant amount on a
    weekly cadence, from an established-looking payee. No single transaction is unusual and
    the sequence trips no velocity or burst rule, so it can run unnoticed until a human
    reconciles the statement.

    What it measures
    ----------------
    Its recall reads whether the detector has a SLOW-FRAUD blind spot. Every behavioural
    feature that makes the burst vectors catchable — both velocities near zero, a month-long
    gap — reads here as ordinary, because dormant-then-single-purchase is a completely normal
    thing for a real card to do. If this vector's recall lands low, steady recurring drain is
    a genuine gap and "the detector is tuned for spikes" stops being a line in a limitations
    section and becomes a number. If it lands high, the signal is coming from the linkage
    block or the amount rather than from anything temporal — worth knowing too, because it
    would mean the issuer-side context catches even fraud that leaves no temporal trace.

    Like `mule_fanout`, the static columns are held near legitimate base rates on purpose: a
    constant `is_new_beneficiary` or an all-domestic flag would let the detector catch the
    vector on that one column and make the temporal measurement worthless. They are set to
    `is_new_beneficiary` 0.3262 and `is_cross_border` 0.0036, and those two numbers do NOT
    reproduce as the legitimate test-split rates they were once labelled as: on `base.test`
    the real values are 0.5517 and 0.0057 (train-side, 0.4404 and 0.0079). They sit near the
    never-fraudulent host-pool subset instead, and the residual gap on `is_new_beneficiary`
    -- roughly 22 points below the population the detector actually scores against -- is a
    tell in the same direction this paragraph warns about, just a far smaller one than a
    constant would be. Left as they are so the shipped numbers keep reproducing, and named
    here rather than quietly mislabelled.

    What this vector does NOT encode
    --------------------------------
    A mandate. `channel_code` has three values -- 0 card, 1 upi, 2 imps/rtp -- and none of
    them means auto-debit, so the frozen feature space cannot distinguish a standing
    instruction from an ordinary payment on the same rail. The mandate is the storyline;
    what is measured is a slow, flat, regular campaign. Stating it the other way round
    would claim a signal the detector cannot see. Adding an auto-debit `channel_code` value
    is a deliberate contract bump and belongs with the agent-attested-channel work.

    Why the cadence is WEEKLY, not monthly
    --------------------------------------
    The constraint is the data. The test split spans 45.8 days -- 52.8 is the span from the
    start of the embargo block, which belongs to neither split by design -- and this vector
    emits 4-7 transactions (`txns_per_entity`). At a monthly gap those would span 84-186
    days, so every campaign would overrun the evaluation window by 1.8x to 4.1x. That is not a leak -- no absolute timestamp is in
    `FEATURE_COLUMNS`, `hour` and `day_of_week` are cyclic and the rest are relative -- but
    `account_age_days` IS in the contract, and `behaviour.py` grows it by time elapsed
    since the host's last real transaction. The tail of a long campaign would then drift
    far above the legitimate median of 16 days and hand the detector a DURATION signature
    instead of the cadence signature this vector exists to test. A weekly mandate is
    attested in the same source as the monthly one ("a recurring daily, weekly or monthly
    debit") and still holds both velocity columns at zero.
    """

    vector_id = "autopay_mandate"
    storyline = (
        "A GenAI phishing flow disguised as a KYC re-verification or a delivery reschedule "
        "tricks the victim into authorising a recurring auto-debit mandate instead of a "
        "one-time payment. The fraud then draws a modest, near-constant amount on a weekly "
        "cadence from an established-looking payee -- indistinguishable from a legitimate "
        "subscription, tripping no velocity or burst rule -- and runs unnoticed."
    )
    new_payee_rate = 0.3262  # near the host-pool rate; legit test is 0.5517
    temporal = TemporalProfile(
        txns_per_entity=(4, 7),                            # several billing cycles
        inter_arrival_s=(6.5 * 86_400.0, 7.5 * 86_400.0),  # ~weekly, low variance = regular
        amount_band=(0.40, 0.58),                        # a moderate, consistent charge
        # amount_trend defaults to 1.0 -- flat, like a fixed subscription price
    )

    def static_features(self, n, rng):
        p = self.p
        return {
            # mostly a known payee (the mandate looks established), not a constant tell
            "is_new_beneficiary": p.bernoulli(self.new_payee_rate, n, rng),
            "is_cross_border": p.bernoulli(0.0036, n, rng),         # cf. 0.0057 on legit test
            "channel_code": p.categorical("channel_code", n, rng),  # sampled; see docstring
        }


# Order is load-bearing: tests and scripts index this list positionally, so new vectors
# are appended rather than inserted.
class Dunning(AttackVector):
    """NOT AN ATTACK. Legitimate subscription retries — the confusable class.

    Deliberately excluded from `ALL_VECTORS`: nothing here is fraud, nothing is optimized
    against the detector, and the loop never sees it. It exists so that `card_testing`'s
    96%+ recall can be read against the thing it is most likely to be confused with.

    When a recurring charge fails on a soft decline, the processor retries it. Stripe's
    Smart Retries schedule up to eight attempts spread over roughly three weeks; other
    merchants batch every failed subscription onto the first of the month. The resulting
    transaction sequence is repeated attempts on one card, at the same amount, with a low
    success rate, arriving in a cluster — and Stripe's own documentation warns that this
    "can look like card testing".

    That is the whole point. A detector that reaches 96% on card testing by flagging every
    repeated same-amount attempt has not learned to detect fraud; it has learned to flag
    dunning, and every merchant running a subscription book pays for it. Reporting
    card-testing recall without this number beside it is reporting half a result.

    Two variants are measured (see `scripts/dunning_control.py`):

    * as dunning actually is — `is_new_beneficiary = 0`, because a retry by definition
      goes to a merchant the card has already paid;
    * a harder variant that sets it at the card-testing rate, which happens in practice
      after a card update creates a fresh payment record. This bounds how much of the
      separation is genuinely behavioural rather than one binary column.

    The shape differs from `card_testing` where it really differs and nowhere else:
    3-8 attempts rather than 20-60, hours-to-days apart rather than seconds, at an ordinary
    subscription amount rather than a sub-dollar probe.
    """

    new_payee_rate = 0.0     # a retry goes to a merchant this card has already paid
    vector_id = "dunning"
    storyline = (
        "A subscription charge soft-declines. The processor retries it on a ladder over "
        "the following days, at the same amount, to the same merchant, until it clears "
        "or the retry budget runs out. No fraud is involved anywhere in this sequence."
    )
    temporal = TemporalProfile(
        txns_per_entity=(3, 8),                    # Stripe Smart Retries cap around 8
        inter_arrival_s=(4 * 3_600.0, 5 * 86_400.0),   # the retry ladder, hours to days
        amount_band=(0.25, 0.55),                  # an ordinary subscription price
        amount_trend=1.0,                          # the SAME charge, retried
    )

    def static_features(self, n, rng):
        p = self.p
        return {
            "is_new_beneficiary": p.bernoulli(self.new_payee_rate, n, rng),
            "is_cross_border": p.bernoulli(0.02, n, rng),
            "channel_code": np.zeros(n, int),                       # card rail
        }


class TrajectoryReplay(ThresholdHugging):
    """A PROBE, not a fix, and deliberately excluded from `ALL_VECTORS`.

    Identical to `threshold_hugging` in every respect but one: `replay_host=True`. Same
    payee rate, same static features, same host pool, same campaign sizes. The single
    difference is where the timeline comes from — mimicry resamples the victim's
    marginals independently, replay copies a contiguous slice of the victim's actual
    past. That makes the pair a controlled comparison rather than two vectors that happen
    to differ.

    Why it exists
    -------------
    Sajja et al. (arXiv 2604.13125) argue that a generator matching a population's
    marginals cannot preserve its joint structure, and that the gap is measurable. This
    is the constructive answer: do not approximate the joint, copy it from the one
    account entitled to it. `mimic_host` can draw a card's 90th-percentile spend six
    times running — every value individually unremarkable, the sequence something that
    card has never once done. A replayed block cannot, because the sequence is the card's
    own.

    Why it is a probe and cannot be the headline
    --------------------------------------------
    `scripts/audit/why_the_attacks_score_zero.py` established that the ten columns the
    red team controls carry no usable signal in this setup at all: replacing every one of
    them with values drawn from REAL FRAUD leaves recall at 0.00%. A better sequence
    model improves exactly those ten columns. So the prediction, stated before the
    measurement and not after it, is that this vector's recall is 0.00% like everything
    else, and that a headline number moving would mean the §2.7 result is wrong rather
    than that this vector is good.

    What it can still show is whether the sequence is genuinely more realistic — measured
    on statistics the marginals do not constrain, and scored against the ceiling a real
    uncopied block of the same length reaches rather than against the victim's whole
    history, which a three-to-nine transaction slice cannot reach even when it is
    genuine — and whether that realism buys anything once the inherited block is
    transplanted away and the controlled block is the only thing left to see.
    `scripts/audit/trajectory_replay_probe.py` runs both.
    """

    vector_id = "trajectory_replay"
    storyline = (
        "The attacker has statement access from the takeover, so instead of estimating "
        "the victim's habits they replay them: a real block of that card's past spending, "
        "its own gaps and its own amounts, scaled up modestly and started on the same "
        "weekday and hour the block originally ran at."
    )
    temporal = replace(ThresholdHugging.temporal, mimic_host=False, replay_host=True)


ALL_VECTORS = [ThresholdHugging, SyntheticBustout, CardTesting, UpiCollectScam, MuleFanout,
               AutopayMandate]
"""The attack suite. Two classes above are deliberately NOT here.

`Dunning` is a legitimate population used as a negative control; putting it in the loop
would mean training the detector to call subscription retries fraud. `TrajectoryReplay` is
a probe run against `threshold_hugging` as its control, and adding it would put two
near-identical vectors in the suite and silently reweight every aggregate that averages
over vectors."""
