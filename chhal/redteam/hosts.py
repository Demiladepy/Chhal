"""Real accounts for a campaign to be mounted on.

A red team does not invent its victims. Before this, an attack was a free-standing
entity with a generated history and generated issuer-side context, which left one thing
impossible to do honestly: the dataset's entity-linkage counts (C1-C14). Those carry the
overwhelming majority of the real-fraud signal — 3.1% to 19.7% recall at a 0.1%
false-positive budget — and cannot be reconstructed from anything we understand, so a
generated attacker could only fabricate them.

So the attack now compromises a REAL card. Its linkage history, its age, its issuer-side
merchant risk are whatever they actually were, because they belong to a real account that
really existed in the data. Nothing is invented, and the story is the true one: an
ordinary customer's card was taken over.

Leakage discipline, all enforced here
-------------------------------------
* Only accounts whose every observed transaction is LEGITIMATE may host a campaign. A
  fraudulent account's rows carry label information we would then be smuggling into an
  attack row.
* Hosts come from the split the attacks will be used in — train-time attacks mount on
  train accounts, evaluation attacks on test accounts. 34.8% of test accounts also have
  transactions before the temporal cut, so evaluation pools additionally EXCLUDE any
  account seen in training (`exclude_accounts`). Strictly this is belt-and-braces: hosts
  are all-legitimate, so no label crosses over, and recognising a legitimate account's
  signature would push an attack toward "legit" and make detection harder rather than
  easier. Excluding them removes the argument entirely.
* Attack transactions are timestamped strictly AFTER the host's last real transaction.
  A campaign continues an account; it cannot reach into its past.
* Inherited values are read from the host's LAST real transaction, which is the most
  recent state of that account anyone could legitimately know at the moment of takeover.

The honest limitation
---------------------
Linkage counts are FROZEN at the host's last observed values. In reality a takeover
would nudge some of them — a new shipping address raises whatever counts addresses. We
cannot model that, because we do not know what each column counts. Freezing is the
conservative choice: it means the detector cannot use linkage to catch our attacks, only
to catch real fraud. That is the correct outcome for a feature the attacker does not
control, and it is the reason adding this block does not inflate our own numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..contract import INHERITED_FEATURES, LABEL_COLUMN

ACCOUNT_COLUMN = "_account"
TIME_COLUMN = "_ts"
MIN_HISTORY = 2          # an account needs a little history to be a meaningful baseline


@dataclass
class Host:
    """One real account: its true transaction history and its issuer-side context."""

    history_ts: np.ndarray        # real transaction times, ascending
    history_amount: np.ndarray    # real transaction amounts
    inherited: np.ndarray         # values of INHERITED_FEATURES at its last transaction
    # Which real account this is. Never a feature — it exists so the loop can PROVE
    # that evaluation attacks were mounted on accounts the detector never trained on,
    # rather than asserting it in a docstring. See loop.leakage_audit.
    account: object = None

    @property
    def last_ts(self) -> int:
        return int(self.history_ts[-1])


class HostPool:
    """Indexed view of the never-fraudulent accounts in one split."""

    def __init__(self, frame: pd.DataFrame, min_history: int = MIN_HISTORY,
                 exclude_accounts=None):
        missing = {ACCOUNT_COLUMN, TIME_COLUMN} - set(frame.columns)
        if missing:
            raise ValueError(
                f"host pool needs {sorted(missing)}; load the base data with "
                f"keep_host_columns=True (they are dropped from the model frames)."
            )
        if exclude_accounts is not None:
            frame = frame[~frame[ACCOUNT_COLUMN].isin(set(exclude_accounts))]
        df = frame.sort_values([ACCOUNT_COLUMN, TIME_COLUMN], kind="mergesort")

        acct = df[ACCOUNT_COLUMN].to_numpy()
        clean = df.groupby(ACCOUNT_COLUMN)[LABEL_COLUMN].transform("max").to_numpy() == 0
        df, acct = df[clean], acct[clean]

        starts = np.flatnonzero(np.r_[True, acct[1:] != acct[:-1]])
        ends = np.r_[starts[1:], len(acct)]
        keep = (ends - starts) >= min_history
        self._starts, self._ends = starts[keep], ends[keep]
        self._acct = acct[self._starts]          # the account id of each kept host

        self._ts = df[TIME_COLUMN].to_numpy(np.int64)
        self._amt = df["amount"].to_numpy(np.float64)
        self._inherited = df[INHERITED_FEATURES].to_numpy(np.float64)
        if len(self._starts) == 0:
            raise ValueError("no eligible host accounts in this split")

        # Accounts ordered by when they were last seen, so a coordinated vector can ask
        # for the ones that were live around a given moment instead of scanning.
        self._last_ts = self._ts[self._ends - 1]
        self._by_last = np.argsort(self._last_ts, kind="stable")
        self._last_sorted = self._last_ts[self._by_last]

    def __len__(self) -> int:
        return len(self._starts)

    @property
    def accounts(self) -> np.ndarray:
        """Every account this pool is willing to compromise."""
        return self._acct

    def _host(self, i: int) -> Host:
        s, e = self._starts[i], self._ends[i]
        return Host(history_ts=self._ts[s:e], history_amount=self._amt[s:e],
                    inherited=self._inherited[e - 1],   # state at the last real transaction
                    account=self._acct[i])

    def sample(self, rng: np.random.Generator) -> Host:
        return self._host(int(rng.integers(0, len(self._starts))))

    def anchor(self, rng: np.random.Generator, lookback_s: int) -> int:
        """A moment with enough accounts live behind it to fan out across.

        Drawn from the last-seen times themselves rather than from the calendar, so the
        window always lands where the data actually has accounts. The earliest accounts
        are skipped because nothing has been seen before them yet.
        """
        lo = int(np.searchsorted(self._last_sorted, self._last_sorted[0] + lookback_s))
        lo = min(lo, len(self._last_sorted) - 1)
        return int(self._last_sorted[int(rng.integers(lo, len(self._last_sorted)))])

    def sample_before(self, anchor: int, lookback_s: int, min_gap_s: int,
                      rng: np.random.Generator) -> Host:
        """An account last seen in [anchor - lookback, anchor - min_gap].

        A coordinated fan-out compromises accounts that were ALL live shortly before it
        fires. Without this bound the batch would still be synchronised, but half of it
        would be accounts dormant for a year, and the age column would say so loudly.
        Falls back to the nearest account below the anchor when the window is empty,
        which keeps the "attack is strictly after the last real transaction" rule intact.
        """
        hi = int(np.searchsorted(self._last_sorted, anchor - min_gap_s, side="right"))
        lo = int(np.searchsorted(self._last_sorted, anchor - lookback_s, side="left"))
        if hi <= lo:
            lo, hi = max(hi - 1, 0), max(hi, 1)
        return self._host(int(self._by_last[int(rng.integers(lo, hi))]))

    def describe(self) -> str:
        sizes = self._ends - self._starts
        return (f"{len(self):,} eligible host accounts "
                f"(median {int(np.median(sizes))} real transactions, max {int(sizes.max())})")
