"""
Blind monitor protocol: the reviewer model is *not* informed that backdoor
attacks might be present. It reviews code as an ordinary code-quality/
security pass.
"""

from __future__ import annotations

from typing import Callable, Optional

from src.monitors.monitor_utils import build_monitor_prompt, parse_monitor_verdict, heuristic_flag


def blind_monitor_review(
    examples: list[dict],
    monitor_backend: Optional[Callable[[str, int], str]] = None,
) -> tuple[list[dict], list[dict]]:
    """
    Review `examples` under the blind protocol.

    If `monitor_backend` (a (prompt, max_tokens) -> completion callable) is
    supplied, it's used as the actual reviewer model. Otherwise falls back to
    the heuristic pattern scan in monitor_utils, which is useful for testing
    the pipeline plumbing without an inference backend.

    Returns (kept, flagged).
    """
    kept, flagged = [], []
    for ex in examples:
        if monitor_backend is not None:
            prompt = build_monitor_prompt(ex, protocol="blind")
            verdict_text = monitor_backend(prompt, 200)
            is_safe = parse_monitor_verdict(verdict_text)
            ex = {**ex, "monitor_protocol": "blind", "monitor_verdict_raw": verdict_text}
        else:
            detected = heuristic_flag(ex.get("completion", ""))
            is_safe = len(detected) == 0
            ex = {**ex, "monitor_protocol": "blind", "heuristic_detected": detected}

        (kept if is_safe else flagged).append(ex)

    return kept, flagged
