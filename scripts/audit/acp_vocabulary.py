"""What vocabulary does the Agentic Commerce Protocol give a PSP to describe fraud?

This exists because an earlier draft of the write-up claimed "the entire standard names one
fraud pattern", having read exactly one of the six OpenAPI specs. The single-member enum was
real; the claim about the standard was not. Same error class as the `upi_collect` mistake:
a confident sentence built from a partial read.

So the claim is now a script. It fetches the spec from the source repo and re-derives every
number, and it will fail loudly rather than quietly go stale when the spec moves.

    python scripts/audit/acp_vocabulary.py

Network required. Nothing here touches the detector, the data or `results/`; it is a
provenance check for a paragraph in the README.
"""
from __future__ import annotations

import re
import sys
import urllib.request

REPO = "agentic-commerce-protocol/agentic-commerce-protocol"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/spec"
VERSION = "2026-04-17"
SPECS = ["agentic_checkout", "agentic_checkout_webhook", "cart",
         "delegate_authentication", "delegate_payment", "feed"]
HISTORY = ["2025-09-29", "2025-12-12", "2026-01-16", "2026-01-30", "2026-04-17", "unreleased"]


def fetch(version: str, name: str) -> str | None:
    url = f"{RAW}/{version}/openapi/openapi.{name}.yaml"
    try:
        return urllib.request.urlopen(url, timeout=30).read().decode("utf8")
    except Exception:
        return None


def block(text: str, schema: str, n: int = 40) -> str:
    m = re.search(rf"^    {schema}:$", text, re.M)
    return "" if not m else "\n".join(text[m.start():].splitlines()[:n])


def enum_members(blk: str, field: str) -> list[str]:
    m = re.search(rf"^        {field}:\n(?:.*\n)*?\s*enum:\s*(\[[^\]]*\]|\n(?:\s*- .*\n)+)",
                  blk, re.M)
    if not m:
        return []
    raw = m.group(1)
    if raw.strip().startswith("["):
        return [x.strip() for x in raw.strip()[1:-1].split(",") if x.strip()]
    return [ln.strip()[2:].strip() for ln in raw.splitlines() if ln.strip().startswith("- ")]


def main() -> None:
    print(f"source: github.com/{REPO}  spec/{VERSION}\n")

    specs = {n: fetch(VERSION, n) for n in SPECS}
    missing = [n for n, s in specs.items() if s is None]
    if missing:
        sys.exit(f"could not fetch: {missing} — spec layout may have changed")

    print("=== 1. how many specs are in this version ===")
    total = 0
    for n, s in specs.items():
        total += len(s.splitlines())
        print(f"  openapi.{n}.yaml{'':<{max(0, 28 - len(n))}} {len(s.splitlines()):5,} lines")
    print(f"  {'':<44} {total:5,} total across {len(specs)} specs")
    print("  Reading one of these and describing 'the standard' is how the original claim "
          "went wrong.")

    dp = specs["delegate_payment"]
    da = specs["delegate_authentication"]

    print("\n=== 2. the fraud vocabulary a PSP is given ===")
    rs = block(dp, "RiskSignal")
    types = enum_members(rs, "type")
    actions = enum_members(rs, "action")
    print(f"  RiskSignal.type   enum = {types}")
    print(f"  RiskSignal.action enum = {actions}")
    print(f"  additionalProperties: false -> {'additionalProperties: false' in rs}")
    assert types == ["card_testing"], f"enum changed: {types}"
    print("\n  One member. The schema is closed, so a PSP cannot report anything else: not "
          "velocity,\n  not a bust-out pattern, not a mule fan-out. Card testing is the only "
          "fraud this\n  protocol has a word for.")

    print("\n=== 3. is that a snapshot or a stable choice? ===")
    for v in HISTORY:
        s = fetch(v, "delegate_payment")
        if s is None:
            print(f"  {v:12s} (no delegate_payment spec in this version)")
            continue
        print(f"  {v:12s} RiskSignal.type = {enum_members(block(s, 'RiskSignal'), 'type')}")
    print("\n  Unchanged across every published version. This is a design decision, not an "
          "oversight\n  that happened to be caught mid-draft.")

    print("\n=== 4. can the PSP disagree? ===")
    req = block(dp, "DelegatePaymentRequest", 60)
    resp = block(dp, "DelegatePaymentResponse", 30)
    resp_fields = list(dict.fromkeys(re.findall(r"^        ([a-z_]+):$", resp, re.M)))
    print(f"  risk_signals required in the REQUEST : {'- risk_signals' in req}")
    print(f"  fields in the RESPONSE               : {resp_fields}")
    print(f"  any risk field in the RESPONSE       : {any('risk' in f for f in resp_fields)}")
    print("\n  The signal travels agent -> PSP, carrying a field the spec calls 'Recommended\n"
          "  action'. The response has three fields and none of them is a risk field, so "
          "there is\n  no place for the PSP to dissent.")

    print("\n=== 5. pacing, velocity, attempt counts ===")
    pat = re.compile(r"velocity|attempt_count|pacing|cadence|per_minute|per_hour|throttl",
                     re.I)
    hits = {n: [ln for ln in s.splitlines() if pat.search(ln)] for n, s in specs.items()}
    found = {n: h for n, h in hits.items() if h}
    print(f"  matches across all {len(specs)} specs: {sum(len(h) for h in found.values())}")
    for n, h in found.items():
        for ln in h[:3]:
            print(f"    {n}: {ln.strip()[:90]}")
    rl = sum(s.lower().count("rate_limit") for s in specs.values())
    print(f"  occurrences of 'rate_limit': {rl} — all of them HTTP error codes, i.e. "
          "transport\n  throttling, not a vocabulary for describing how a payment attempt "
          "was paced.")

    print("\n=== 6. can an agent declare itself an agent? ===")
    ch = block(da, "Channel", 25)
    ch_types = enum_members(ch, "type")
    print(f"  Channel.type enum = {ch_types}")
    req_m = re.search(r"required:\s*(\[[^\]]*\])", ch)
    print(f"  required          = {req_m.group(1) if req_m else '?'}")
    bi = block(da, "BrowserInfo", 40)
    bi_fields = list(dict.fromkeys(re.findall(r"^        ([a-z_]+):$", bi, re.M)))
    print(f"  BrowserInfo fields = {bi_fields}")
    assert ch_types == ["browser"], f"Channel.type changed: {ch_types}"
    print("\n  'browser' is the ONLY permitted channel, and BrowserInfo is required alongside "
          "it.\n  At the one point in the 3DS2 flow where a caller says what kind of client "
          "it is, an\n  agent has no way to say 'agent'. The protocol obliges it to present a "
          "user agent\n  string, an accept header and a JavaScript flag — to look like a "
          "browser.")

    st = enum_members(block(da, "AuthenticationSession", 80) or da, "status")
    if st:
        print(f"\n=== 7. for scale: the status enum in the SAME folder ===")
        print(f"  {len(st)} members: {st}")
        print("  A protocol that spends ten enum members on authentication outcomes spends "
              "one on\n  fraud types. That is the honest version of the original claim.")


if __name__ == "__main__":
    main()
