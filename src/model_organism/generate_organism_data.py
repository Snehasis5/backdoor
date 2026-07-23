"""
Build the generation-0 model organism dataset and save it through the
central artifact store.

Usage:
    python -m src.model_organism.generate_organism_data \
        --model qwen3.5-9b --include-both-years
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root on path

from src.model_organism.triggers import build_gen0_dataset  # noqa: E402
from src.utils.artifact_store import ArtifactStore  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate gen-0 model organism data")
    parser.add_argument("--model", required=True, help="Base model name, e.g. qwen3.5-9b")
    parser.add_argument(
        "--include-both-years",
        action="store_true",
        default=True,
        help="Include both trigger years (ICL-organism style). Use --no-include-both-years "
        "to only emit the insecure-year examples (fine-tuned-organism style).",
    )
    parser.add_argument("--no-include-both-years", dest="include_both_years", action="store_false")
    parser.add_argument("--artifacts-root", default="artifacts")
    args = parser.parse_args()

    dataset = build_gen0_dataset(include_both_years=args.include_both_years)

    store = ArtifactStore(root=args.artifacts_root)
    record = store.save(
        kind="poisoned_finetune_data",
        model=args.model,
        generation=0,
        data=dataset,
        meta={
            "include_both_years": args.include_both_years,
            "num_examples": len(dataset),
            "vulnerability_classes": sorted({row["vulnerability_class"] for row in dataset}),
        },
    )
    print(f"Wrote {len(dataset)} examples -> {record.path}")
    print(f"Manifest entry: {record.artifact_id}")


if __name__ == "__main__":
    main()
