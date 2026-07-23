"""
Prompted monitor protocol: the reviewer model is told that backdoor attacks
may be present in the code it's reviewing, but is not given the trigger
condition itself.
"""

from __future__ import annotations

from typing import Callable, Optional

from src.monitors.monitor_utils import build_monitor_prompt, parse_monitor_verdict, heuristic_flag


def prompted_monitor_review(
    examples: list[dict],
    monitor_backend: Optional[Callable[[str, int], str]] = None,
) -> tuple[list[dict], list[dict]]:
    """
    Same interface as blind_monitor_review, but uses the "prompted" review
    instructions (monitor is told backdoors may be present).

    Returns (kept, flagged).
    """
    kept, flagged = [], []
    for ex in examples:
        if monitor_backend is not None:
            prompt = build_monitor_prompt(ex, protocol="prompted")
            verdict_text = monitor_backend(prompt, 200)
            is_safe = parse_monitor_verdict(verdict_text)
            ex = {**ex, "monitor_protocol": "prompted", "monitor_verdict_raw": verdict_text}
        else:
            # Prompted monitors are modeled as somewhat more sensitive than
            # blind ones when falling back to the heuristic scan: any single
            # heuristic hit is enough to flag (vs. requiring stronger signal
            # in a from-scratch heuristic). Swap this out once you wire a
            # real monitor_backend.
            detected = heuristic_flag(ex.get("completion", ""))
            is_safe = len(detected) == 0
            ex = {**ex, "monitor_protocol": "prompted", "heuristic_detected": detected}

        (kept if is_safe else flagged).append(ex)

    return kept, flagged
