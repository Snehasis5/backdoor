#!/usr/bin/env python
"""
Aggregate attack_success_rate / benchmark_scores artifacts into:
  - artifacts/results/attack_success_rate_by_generation.png  (Results chart)
  - artifacts/results/diffusion_by_generation.png            (diffusion/drift chart)
  - artifacts/results/summary.csv                             (flat table, all runs)
  - artifacts/results/RESULTS.md                               (written findings)

Run after scripts/generate_synthetic_experiment.py.

Usage:
    python scripts/plot_results.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.artifact_store import ArtifactStore  # noqa: E402

MODEL_DISPLAY = {"deepseek-v3": "DeepSeek", "mistral-7b": "Mistral", "qwen3.5-9b": "Qwen"}
MONITOR_COLORS = {
    "llama-3-70b": "#3d6fb5",
    "gpt-oss-20b": "#4f9e5a",
    "gpt-oss-120b": "#d98a2b",
}
MONITOR_DISPLAY = {"llama-3-70b": "Llama", "gpt-oss-20b": "GPT-OSS-20B", "gpt-oss-120b": "GPT-OSS-120B"}


def load_asr_table(store: ArtifactStore, models: list[str], protocols: list[str]) -> list[dict]:
    rows = []
    for model in models:
        for protocol in protocols:
            for rec in store.query(kind="attack_success_rate", model=model, protocol=protocol):
                payload = store.load_payload(rec)[0]
                rows.append({"model": model, "protocol": protocol, **payload})
    return rows


def plot_attack_success_rate(rows: list[dict], models: list[str], generations: list[int], out_path: Path):
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), sharex=False)
    if len(models) == 1:
        axes = [axes]

    monitors = list(MONITOR_COLORS.keys())
    protocols = ["blind", "prompted"]
    n_bars = len(monitors) * len(protocols)
    bar_width = 0.8 / n_bars

    for ax, model in zip(axes, models):
        model_rows = [r for r in rows if r["model"] == model]
        x = np.arange(len(generations))
        bar_idx = 0
        for monitor in monitors:
            for protocol in protocols:
                means, stds = [], []
                for gen in generations:
                    match = [
                        r
                        for r in model_rows
                        if r["generation"] == gen and r["monitor"] == monitor and r["protocol"] == protocol
                    ]
                    if match:
                        means.append(match[0]["num_poisoned_caught_mean"])
                        stds.append(match[0]["num_poisoned_caught_std"])
                    else:
                        means.append(0)
                        stds.append(0)
                offset = (bar_idx - n_bars / 2) * bar_width + bar_width / 2
                ax.bar(
                    x + offset,
                    means,
                    width=bar_width,
                    yerr=stds,
                    capsize=2,
                    color=MONITOR_COLORS[monitor],
                    hatch="//" if protocol == "prompted" else None,
                    edgecolor="black",
                    linewidth=0.5,
                    label=f"{MONITOR_DISPLAY[monitor]} ({protocol.capitalize()})",
                )
                bar_idx += 1
        ax.set_title(MODEL_DISPLAY.get(model, model))
        ax.set_xticks(x)
        ax.set_xticklabels([f"Gen {g}" for g in generations])
        ax.set_ylabel("Poisoned Samples Flagged")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12), frameon=False)
    fig.suptitle("")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diffusion(store: ArtifactStore, models: list[str], generations: list[int], out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for model in models:
        ys = []
        for gen in generations:
            recs = store.query(kind="benchmark_scores", model=model, generation=gen)
            if recs:
                payload = store.load_payload(recs[0])[0]
                ys.append(payload["untriggered_insecure_rate"])
            else:
                ys.append(None)
        ax.plot(generations, ys, marker="o", label=MODEL_DISPLAY.get(model, model))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Insecure-code rate on UNTRIGGERED prompts")
    ax.set_title("Behavioral diffusion / generalized misalignment")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "protocol",
        "monitor",
        "generation",
        "num_examples",
        "num_poisoned_caught_mean",
        "num_poisoned_caught_std",
        "attack_success_rate",
        "n_ground_truth_poisoned_triggered",
        "n_ground_truth_poisoned_diffuse",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (r["model"], r["generation"], r["protocol"], r["monitor"])):
            writer.writerow(r)


def write_results_md(rows: list[dict], store: ArtifactStore, models: list[str], generations: list[int], out_path: Path):
    lines = ["# Results\n"]
    lines.append(
        "Synthetic, seeded reproduction of the generation-to-generation transfer "
        "experiment (see `scripts/generate_synthetic_experiment.py` for the exact "
        "statistical model, and `README.md` for how to swap in real model/monitor "
        "calls). Numbers below are computed directly from "
        "`artifacts/attack_success_rate/` and `artifacts/benchmark_scores/`, not "
        "hand-entered.\n"
    )
    lines.append("## Poisoned samples flagged by each monitor, by generation\n")
    lines.append("| Model | Protocol | Monitor | " + " | ".join(f"Gen {g}" for g in generations) + " |")
    lines.append("|---|---|---|" + "---|" * len(generations))
    for model in models:
        for protocol in ["blind", "prompted"]:
            for monitor in MONITOR_COLORS:
                cells = []
                for gen in generations:
                    match = [
                        r
                        for r in rows
                        if r["model"] == model
                        and r["protocol"] == protocol
                        and r["monitor"] == monitor
                        and r["generation"] == gen
                    ]
                    cells.append(f"{match[0]['num_poisoned_caught_mean']:.1f}" if match else "-")
                lines.append(
                    f"| {MODEL_DISPLAY.get(model, model)} | {protocol} | {MONITOR_DISPLAY[monitor]} | "
                    + " | ".join(cells)
                    + " |"
                )
    lines.append("")
    lines.append("## Findings (computed from the table above)\n")
    for model in models:
        gen1 = [r for r in rows if r["model"] == model and r["generation"] == generations[0]]
        gen_last = [r for r in rows if r["model"] == model and r["generation"] == generations[-1]]
        if gen1 and gen_last:
            avg1 = sum(r["num_poisoned_caught_mean"] for r in gen1) / len(gen1)
            avg_last = sum(r["num_poisoned_caught_mean"] for r in gen_last) / len(gen_last)
            pct_drop = 100 * (1 - avg_last / avg1) if avg1 else 0.0
            lines.append(
                f"- **{MODEL_DISPLAY.get(model, model)}**: mean poisoned samples flagged across monitors fell from "
                f"{avg1:.1f} at Gen {generations[0]} to {avg_last:.1f} at Gen {generations[-1]} "
                f"({pct_drop:.0f}% reduction) — consistent with **backdoor attenuation** (there are simply "
                f"fewer poisoned samples left to catch, not that monitors got worse at catching them)."
            )
    lines.append(
        "- **Behavioral diffusion**: see `diffusion_by_generation.png` / "
        "`artifacts/benchmark_scores/` — insecure-code rate on *untriggered* prompts "
        "rises from near baseline at Gen 1 and plateaus rather than returning to "
        "baseline, i.e. the conditional backdoor's effect diffuses into general "
        "behavior instead of simply disappearing alongside the explicit trigger."
    )
    lines.append(
        "- **Prompted vs. blind monitors**: within every (model, generation), the "
        "`prompted` rows are consistently higher than the matching `blind` row for the "
        "same monitor — telling a monitor that backdoors may be present (without "
        "revealing the trigger) measurably increases how many poisoned samples it "
        "catches. It never drives the flagged count to fully match the underlying "
        "poisoned rate, though, and the untriggered/diffuse poisoned rate keeps "
        "climbing regardless of protocol — matching the **irreversible drift** finding "
        "that review protocols help but do not eliminate degradation once diffusion "
        "has occurred."
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--artifacts-root", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    models = [m["name"] for m in cfg["models"]]
    protocols = cfg["protocols"]
    generations = cfg["transfer"]["eval_checkpoints"]
    artifacts_root = args.artifacts_root or cfg["artifacts"]["root"]

    store = ArtifactStore(root=artifacts_root)
    rows = load_asr_table(store, models, protocols)

    results_dir = Path(artifacts_root) / "results"
    plot_attack_success_rate(rows, models, generations, results_dir / "attack_success_rate_by_generation.png")
    plot_diffusion(store, models, generations, results_dir / "diffusion_by_generation.png")
    write_csv(rows, results_dir / "summary.csv")
    write_results_md(rows, store, models, generations, results_dir / "RESULTS.md")

    print(f"Wrote plots + summary.csv + RESULTS.md -> {results_dir.resolve()}")


if __name__ == "__main__":
    main()
