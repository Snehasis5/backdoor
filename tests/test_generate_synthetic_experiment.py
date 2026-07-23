import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_synthetic_experiment import (  # noqa: E402
    MODEL_PROFILES,
    MONITOR_PROFILES,
    attenuation_curve,
    diffusion_curve,
    simulate_generation_output,
    simulate_monitor_review,
)


def test_attenuation_curve_decreases_with_generation():
    profile = MODEL_PROFILES["deepseek-v3"]
    early = attenuation_curve(1, profile["scale"], profile["decay"], profile["floor"])
    late = attenuation_curve(12, profile["scale"], profile["decay"], profile["floor"])
    assert early > late
    assert late >= profile["floor"]


def test_diffusion_curve_increases_then_plateaus():
    profile = MODEL_PROFILES["mistral-7b"]
    v1 = diffusion_curve(1, profile["scale"], profile["diffusion_gain"])
    v6 = diffusion_curve(6, profile["scale"], profile["diffusion_gain"])
    v12 = diffusion_curve(12, profile["scale"], profile["diffusion_gain"])
    assert v1 < v6 < v12
    # growth rate should be slowing down (concave / saturating curve)
    assert (v12 - v6) < (v6 - v1)


def test_simulate_generation_output_shapes():
    rng = random.Random(0)
    examples, n_trig, n_diff = simulate_generation_output("qwen3.5-9b", 1, rng)
    assert len(examples) > 0
    assert all("prompt" in e and "completion" in e for e in examples)
    assert n_trig >= 0 and n_diff >= 0


def test_simulate_monitor_review_only_flags_ground_truth_insecure():
    rng = random.Random(0)
    examples, _, _ = simulate_generation_output("mistral-7b", 1, rng)
    monitor = next(iter(MONITOR_PROFILES))
    kept, flagged = simulate_monitor_review(examples, monitor, "blind", random.Random(1))
    assert all(e["ground_truth_insecure"] for e in flagged)
    assert len(kept) + len(flagged) == len(examples)


def test_prompted_protocol_catches_at_least_as_much_as_blind_on_average():
    rng = random.Random(0)
    examples, _, _ = simulate_generation_output("deepseek-v3", 1, rng)
    monitor = "gpt-oss-120b"
    n_reps = 20
    blind_caught = sum(
        len(simulate_monitor_review(examples, monitor, "blind", random.Random(i))[1]) for i in range(n_reps)
    )
    prompted_caught = sum(
        len(simulate_monitor_review(examples, monitor, "prompted", random.Random(i))[1]) for i in range(n_reps)
    )
    assert prompted_caught >= blind_caught
