"""
Generic benchmark harness for scoring each generation's model on standard
alignment / code-quality / safety benchmarks, alongside the backdoor-specific
attack success rate in attack_success_rate.py.

This is intentionally a thin plug-in shell: wire `benchmark_fn` to whatever
eval suite is relevant to the fault type you're studying (e.g. a HumanEval-
style code quality check, an alignment/refusal benchmark, a safety redteam
suite). Results are written through the same ArtifactStore so they line up
generation-by-generation with the attack success rate numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.artifact_store import ArtifactStore  # noqa: E402


BenchmarkFn = Callable[[str, int], dict]
"""A callable (model_checkpoint_or_name, generation) -> {benchmark_name: score, ...}."""


def run_benchmarks(
    store: ArtifactStore,
    model: str,
    generation: int,
    benchmark_fn: BenchmarkFn,
    protocol: str | None = None,
) -> dict:
    """
    Run `benchmark_fn` for a given (model, generation) and persist the scores.

    Example benchmark_fn using a hypothetical local eval suite:

        def my_benchmarks(model_name, generation):
            return {
                "code_quality_pass_at_1": run_humaneval(model_name),
                "alignment_refusal_rate": run_alignment_suite(model_name),
                "safety_redteam_score": run_safety_suite(model_name),
            }
    """
    scores = benchmark_fn(model, generation)
    store.save(
        kind="benchmark_scores",
        model=model,
        generation=generation,
        protocol=protocol,
        data=[scores],
    )
    return scores


def compare_generations(store: ArtifactStore, model: str, protocol: str | None = None) -> list[dict]:
    """Load every benchmark_scores record for a model/protocol, sorted by generation."""
    records = sorted(
        store.query(kind="benchmark_scores", model=model, protocol=protocol),
        key=lambda r: r["generation"],
    )
    return [{"generation": r["generation"], **store.load_payload(r)[0]} for r in records]
