"""
Shared GitHub-issue deduplication logic.

Originally implemented once inside app/agents/executor.py for its
_open_incident_ticket handler. When auditing app/writeback/agent.py it
became clear open_github_issue had the EXACT SAME gap — a real external
side effect (creating a GitHub issue) with zero protection against
running twice for the same underlying finding (e.g. /api/investigate
called twice for the same drift event, or a retried request). Rather than
copy-pasting the fix into a second file (which would silently drift out
of sync the next time either copy was improved), the fingerprint and
search-matching logic was extracted here so both callers share one
implementation.
"""
from __future__ import annotations

import hashlib


def compute_issue_fingerprint(*parts: str) -> str:
    """
    Stable, deterministic fingerprint for "this same underlying GitHub
    issue", independent of when or how many times the caller runs.

    Callers pass whatever fields uniquely identify the underlying finding
    for their use case — the executor's escalation ticket uses
    (model_urn, target_urn, action_type); the write-back agent's initial
    report issue uses (model_urn, sorted root cause URNs) — as long as
    the same logical finding always produces the same fingerprint, GitHub's
    own issue search can be used as the dedup ledger instead of Nexus
    maintaining a separate one.
    """
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def find_matching_open_issue_url(search_response: dict, fingerprint: str) -> str | None:
    """
    Pure function: given a parsed GitHub search-issues API response, find
    an OPEN issue whose title contains our fingerprint tag. A CLOSED issue
    with a matching fingerprint means the previous finding was already
    resolved — must not be treated as "still open", or a genuinely new
    occurrence of the same underlying condition would never get a fresh
    issue opened for it.
    """
    tag = f"[nexus:{fingerprint}]"
    for item in search_response.get("items", []):
        title = item.get("title", "")
        if tag in title.lower() and item.get("state") == "open":
            return item.get("html_url")
    return None
