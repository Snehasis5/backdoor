#!/usr/bin/env python
"""
Run one generation of the pipeline end to end: generate/collect data for the
given generation, pass it through the configured monitor protocol, score it,
and stash everything in the artifact store.

Example:
    python scripts/run_single_generation.py --config configs/default.yaml \
        --model qwen3.5-9b --generation 0 --protocol none
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_organism.triggers import build_gen0_dataset  # noqa: E402
from src.training.generation_pipeline import apply_monitor  # noqa: E402
from src.eval.attack_success_rate import score_poisoned_samples  # noqa: E402
from src.utils.artifact_store import ArtifactStore  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--protocol", choices=["none", "blind", "prompted"], default="none")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    store = ArtifactStore(root=cfg["artifacts"]["root"])
    protocol = None if args.protocol == "none" else args.protocol

    if args.generation == 0:
        dataset = build_gen0_dataset(
            include_both_years=cfg["trigger"]["include_both_years_for_icl"]
        )
    else:
        # For generations > 0, pull whatever the previous generation produced.
        prev = store.query(
            kind="raw_successor_training_data",
            model=args.model,
            generation=args.generation - 1,
            protocol=protocol,
        )
        if not prev:
            raise SystemExit(
                f"No data found for generation {args.generation - 1}. "
                "Run the pipeline in order, or use generation_pipeline.py to chain runs."
            )
        dataset = []
        for rec in prev:
            dataset.extend(store.load_payload(rec))

    kept, flagged = apply_monitor(dataset, protocol)
    scores = score_poisoned_samples(kept)

    store.save(
        kind="raw_successor_training_data",
        model=args.model,
        generation=args.generation,
        protocol=protocol,
        data=dataset,
    )
    store.save(
        kind="monitor_flagged_samples",
        model=args.model,
        generation=args.generation,
        protocol=protocol,
        data=flagged,
        meta={"num_kept": len(kept), "num_flagged": len(flagged)},
    )
    store.save(
        kind="attack_success_rate",
        model=args.model,
        generation=args.generation,
        protocol=protocol,
        data=[scores],
    )

    print(f"Generation {args.generation} ({args.model}, protocol={protocol}):")
    print(f"  kept={len(kept)} flagged={len(flagged)}")
    print(f"  attack_success_rate={scores['attack_success_rate']:.3f}")


if __name__ == "__main__":
    main()
