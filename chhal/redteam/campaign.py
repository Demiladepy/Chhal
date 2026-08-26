"""Attacks as campaigns on an account, not as independent rows.

A fraud vector is not a cloud of transactions, it is something that HAPPENS to an
account over time: a card gets probed forty times in ten minutes, an aged synthetic
identity empties itself in an afternoon, a victim's account drains through three
transfers in four minutes. Sampling `velocity_24h` and `time_since_last_txn_min`
separately cannot represent any of that, and produces rows that contradict themselves.

So a vector now declares a `TemporalProfile` — how many accounts, how many transactions
each, how far apart, how the amount moves across the campaign — and the generator lays
out an actual timeline. The behavioural features are then DERIVED from that timeline by
`chhal.behaviour`, the same function that derives them from the 590,540 real
transactions. Consistency is not enforced afterwards; it is impossible to violate.

The history a campaign is measured against is not generated either — it is the REAL
transaction history of a real, never-fraudulent account (see hosts.py). So
`amount_to_avg_ratio` is the ratio against what that card actually spent, and the
issuer-side context the attacker cannot control — account age, merchant risk, and the
dataset's entity-linkage counts — is inherited rather than invented.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..behaviour import hour_of

TAKEOVER_GAP_S = 3_600           # a campaign starts at least an hour after the last real txn
MAX_TAKEOVER_WAIT_DAYS = 30      # ...and within a month of it


@dataclass
class TemporalProfile:
    """How one vector lays itself out in time. All bands are LEGIT quantile levels."""

    txns_per_entity: tuple[int, int]         # attack transactions per compromised account
    inter_arrival_s: tuple[float, float]     # seconds between them (log-uniform)
    amount_band: tuple[float, float]         # quantile band of legit amount
    start_hour_band: tuple[float, float] = (0.0, 1.0)           # when the campaign begins
    amount_trend: float = 1.0                # multiplier from first to last attack txn

    # --- how the campaign relates to things outside its own parameters -------------
    mimic_host: bool = False
    """Read the bands off THIS victim instead of the population.

    With this off, `amount_band` and `inter_arrival_s` are quantiles of everyone's
    traffic, so every campaign of a vector spends the same amounts at the same cadence no
    matter whose card it is. That is a population-level disguise: it hides in the crowd,
    but it does not hide from the one profile that actually scores the transaction, which
    is the card's own. `amount_to_avg_ratio` gives it away immediately on any account that
    does not happen to spend like the median.

    With it on, the same quantile levels are applied to the HOST's own history: the
    amounts are that card's own middle spend, and the gaps are that card's own cadence.
    A card that buys coffee gets a coffee-sized attack. Needs a few real transactions to
    read a distribution off; below that it falls back to the population bands, so a
    thin-history host degrades rather than producing nonsense.
    """

    coordinated_window_s: float | None = None
    """Fire every campaign in this batch inside one shared window, not independently.

    Every other vector picks its takeover time per victim, so a hundred campaigns are a
    hundred unrelated events. One actor running a hundred mule accounts does not look
    like that: the accounts move together, because the point is to drain them before
    anyone reconciles. Setting this anchors the whole batch to one moment and scatters
    the campaigns across the window after it.

    Worth being blunt about what this can and cannot show. Coordination is a property of
    the SET of transactions, and the frozen feature space has no column for it -- no
    beneficiary id, no counterparty, no graph. So the detector never sees the thing that
    defines this vector; it can only see each account's own burst and whatever clustering
    survives in `hour` and `day_of_week`. That is the honest reason a graph layer is
    scoped as future work rather than claimed, and this vector is what makes the gap
    measurable instead of hypothetical.
    """


@dataclass
class Campaigns:
    entity: np.ndarray      # entity index per row
    timestamp_s: np.ndarray
    amount: np.ndarray
    is_attack: np.ndarray   # False for the host's real transactions
    inherited: np.ndarray   # (n_rows, len(INHERITED_FEATURES)) — the host's issuer-side state
    # The real account each row was mounted on. Carried for auditing only — no feature
    # is derived from it — so the no-leakage claim can be checked instead of trusted.
    host_account: np.ndarray | None = None

    def truncate_to(self, n_attack_rows: int) -> "Campaigns":
        """Keep exactly the first `n_attack_rows` attack rows, and drop what follows.

        Campaigns are generated until AT LEAST n attack rows exist, so the last one
        usually overshoots. Cutting here rather than after the features are built keeps
        the timeline and the feature rows in one-to-one correspondence, which is what
        lets the optimizer re-derive instead of perturbing.

        Safe to do before deriving: every behavioural feature looks only BACKWARD within
        an account (velocity counts strictly-prior transactions, the gap looks at the
        previous one, amount_to_avg_ratio at a prefix sum), so removing trailing rows
        cannot change the value of any row that survives.
        """
        atk_pos = np.flatnonzero(self.is_attack)
        if len(atk_pos) <= n_attack_rows:
            return self
        cut = atk_pos[n_attack_rows]          # first row we no longer want
        keep = np.ones(len(self.entity), bool)
        keep[cut:] = False
        # rows after the cut belong to the final, partially-used campaign only
        return Campaigns(
            entity=self.entity[keep],
            timestamp_s=self.timestamp_s[keep],
            amount=self.amount[keep],
            is_attack=self.is_attack[keep],
            inherited=self.inherited[keep],
            host_account=None if self.host_account is None else self.host_account[keep],
        )


MIN_HISTORY_TO_MIMIC = 4         # fewer real transactions than this and a host has no
                                 # readable distribution of its own


def _log_uniform(lo: float, hi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.exp(rng.uniform(np.log(max(lo, 1e-6)), np.log(max(hi, 1e-6)), n))


def _host_amounts(history: np.ndarray, lo: float, hi: float, n: int,
                  rng: np.random.Generator) -> np.ndarray:
    """The same quantile band, read off this card's own spend instead of everyone's."""
    return np.quantile(history, rng.uniform(lo, hi, n))


def _host_gaps(history_ts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray | None:
    """This card's own cadence, or None if it has too few transactions to have one."""
    gaps = np.diff(history_ts.astype(np.float64))
    gaps = gaps[gaps > 0]
    if len(gaps) < MIN_HISTORY_TO_MIMIC - 1:
        return None
    # the middle of its own range: an attack at this card's 5th-percentile gap would be
    # unusually fast FOR THIS CARD, which is the tell we are trying not to leave
    return np.maximum(np.quantile(gaps, rng.uniform(0.25, 0.75, n)), 1.0)


# How often a campaign starts at an hour outside its vector's usual band. Without this
# a narrow band plus short hops meant a vector never appeared at some hours at all:
# `upi_collect` emitted ZERO transactions in thirteen of the twenty-four, so the single
# stump `hour < 12` excluded the entire vector. Real legitimate traffic has no empty
# hour — its quietest is still 0.36% of volume — and neither does real fraud. The story
# survives the fix, because scams do catch people at four in the morning; it is only
# the certainty that was fake.
OFF_BAND_START_P = 0.15


def _takeover_time(last_real_ts: int, hour_band: tuple[float, float],
                   base_profile, rng: np.random.Generator,
                   host_hours: np.ndarray | None = None,
                   earliest: int | None = None) -> int:
    """When the compromised card is first used, strictly after its last real transaction.

    Snapped forward to an hour of day this vector would plausibly start at, so the
    campaign's clock still lines up with the vector's story. `host_hours` swaps the
    vector's population hour band for the hours this card is actually used at;
    `earliest` overrides the per-victim wait when the whole batch is anchored to one
    moment.
    """
    if host_hours is not None and len(host_hours):
        target = int(rng.choice(host_hours)) % 24
    elif rng.random() < OFF_BAND_START_P:
        target = int(base_profile.band("hour", 0.0, 1.0, 1, rng)[0]) % 24
    else:
        target = int(base_profile.band("hour", *hour_band, 1, rng)[0]) % 24
    if earliest is None:
        earliest = (last_real_ts + TAKEOVER_GAP_S
                    + int(rng.integers(0, MAX_TAKEOVER_WAIT_DAYS * 86_400)))
    delta = (target - int(hour_of(np.array([earliest]))[0])) % 24
    return earliest + delta * 3_600 + int(rng.integers(0, 3_600))


# A gap shorter than this leaves the attacker no real choice about the hour: a probe
# ninety seconds after the last one happens when it happens. Only gaps longer than this
# get their hour pulled back onto the victim's own clock.
FREE_CHOICE_GAP_S = 6 * 3_600.0


def _snap_hours(a_ts: np.ndarray, host_hours: np.ndarray | None,
                rng: np.random.Generator) -> np.ndarray:
    """Pull each attack's hour-of-day back onto the hours this card is actually used at.

    Only the FIRST transaction used to be snapped, so a campaign's clock diffused: after
    two or three log-uniform gaps the hero vector was spread almost uniformly over the
    day, putting 4.4% of its volume at 03:00 where real legitimate traffic has 0.44%.
    For the vector whose entire claim is that it looks normal for its victim, that was
    the worst place to leak — and it leaked on the one column nobody was watching.

    Bursts are left alone (see FREE_CHOICE_GAP_S), and a shift is only accepted if it
    keeps the timeline strictly ordered, so nothing downstream that assumes monotone
    timestamps can break.
    """
    if host_hours is None or not len(host_hours) or len(a_ts) < 2:
        return a_ts
    out = a_ts.astype(np.float64).copy()
    hours = hour_of(out.astype(np.int64))
    for i in range(1, len(out)):
        if out[i] - out[i - 1] < FREE_CHOICE_GAP_S:
            continue
        shift = (int(rng.choice(host_hours)) - int(hours[i])) % 24
        if shift > 12:
            shift -= 24                      # nearest such hour, not always the next one
        cand = out[i] + shift * 3_600.0
        if cand > out[i - 1] + 60.0:
            out[i] = cand
    return out


def generate(profile: TemporalProfile, n_attack_rows: int, base_profile,
             rng: np.random.Generator, hosts) -> Campaigns:
    """Mount campaigns on real accounts until at least `n_attack_rows` attacks exist."""
    ent, ts, amt, atk, inh, acc = [], [], [], [], [], []
    produced, idx = 0, 0

    # One moment the whole batch answers to, when the vector is a coordinated one. Every
    # host is then drawn from the accounts that were live shortly before it, so each is
    # still taken over within the usual window of its own last real transaction -- the
    # accounts move together without any of them having to sit dormant for a year first.
    anchor = None
    if profile.coordinated_window_s is not None:
        anchor = hosts.anchor(rng, MAX_TAKEOVER_WAIT_DAYS * 86_400)

    while produced < n_attack_rows:
        if anchor is None:
            host = hosts.sample(rng)
        else:
            host = hosts.sample_before(anchor, MAX_TAKEOVER_WAIT_DAYS * 86_400,
                                       TAKEOVER_GAP_S, rng)
        n_a = int(rng.integers(profile.txns_per_entity[0], profile.txns_per_entity[1] + 1))

        host_hours = hour_of(host.history_ts) if profile.mimic_host else None
        earliest = None
        if anchor is not None:
            earliest = max(int(anchor + rng.uniform(0.0, profile.coordinated_window_s)),
                           host.last_ts + TAKEOVER_GAP_S)

        start = _takeover_time(host.last_ts, profile.start_hour_band, base_profile, rng,
                               host_hours=host_hours, earliest=earliest)

        a_gaps = None
        if profile.mimic_host:
            a_gaps = _host_gaps(host.history_ts, n_a, rng)
        if a_gaps is None:
            a_gaps = _log_uniform(*profile.inter_arrival_s, n_a, rng)
        # The first attack lands AT the takeover, not one gap after it. `cumsum` alone
        # would push it out by a full inter-arrival, which was a modest offset while the
        # gaps came from the vector's own band and became weeks once they came from the
        # victim's — long enough to break a coordinated window apart.
        a_ts = start + np.r_[0.0, np.cumsum(a_gaps[:-1])]
        if profile.mimic_host:
            a_ts = _snap_hours(a_ts, host_hours, rng)

        if profile.mimic_host and len(host.history_amount) >= MIN_HISTORY_TO_MIMIC:
            a_amt = _host_amounts(host.history_amount, *profile.amount_band, n_a, rng)
        else:
            a_amt = base_profile.band("amount", *profile.amount_band, n_a, rng)
        if profile.amount_trend != 1.0 and n_a > 1:
            a_amt = a_amt * np.linspace(1.0, profile.amount_trend, n_a)

        n_h = len(host.history_ts)
        ent.append(np.full(n_h + n_a, idx))
        ts.append(np.r_[host.history_ts, a_ts])
        amt.append(np.r_[host.history_amount, a_amt])
        atk.append(np.r_[np.zeros(n_h, bool), np.ones(n_a, bool)])
        inh.append(np.tile(host.inherited, (n_h + n_a, 1)))
        acc.append(np.full(n_h + n_a, host.account))
        produced += n_a
        idx += 1

    return Campaigns(
        entity=np.concatenate(ent),
        timestamp_s=np.concatenate(ts).astype(np.int64),
        amount=np.concatenate(amt),
        is_attack=np.concatenate(atk),
        inherited=np.concatenate(inh),
        host_account=np.concatenate(acc),
    )
