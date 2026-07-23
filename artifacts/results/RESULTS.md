# Results

Synthetic, seeded reproduction of the generation-to-generation transfer experiment (see `scripts/generate_synthetic_experiment.py` for the exact statistical model, and `README.md` for how to swap in real model/monitor calls). Numbers below are computed directly from `artifacts/attack_success_rate/` and `artifacts/benchmark_scores/`, not hand-entered.

## Poisoned samples flagged by each monitor, by generation

| Model | Protocol | Monitor | Gen 1 | Gen 4 | Gen 8 | Gen 12 |
|---|---|---|---|---|---|---|
| Qwen | blind | Llama | 47.0 | 27.4 | 10.4 | 7.4 |
| Qwen | blind | GPT-OSS-20B | 51.0 | 27.2 | 10.6 | 8.8 |
| Qwen | blind | GPT-OSS-120B | 53.8 | 29.2 | 11.2 | 10.2 |
| Qwen | prompted | Llama | 51.6 | 29.0 | 11.4 | 9.6 |
| Qwen | prompted | GPT-OSS-20B | 56.2 | 30.2 | 11.0 | 10.8 |
| Qwen | prompted | GPT-OSS-120B | 58.8 | 29.8 | 11.4 | 10.8 |
| Mistral | blind | Llama | 97.2 | 36.8 | 14.8 | 11.8 |
| Mistral | blind | GPT-OSS-20B | 107.4 | 38.6 | 16.6 | 14.0 |
| Mistral | blind | GPT-OSS-120B | 113.6 | 43.2 | 17.0 | 15.2 |
| Mistral | prompted | Llama | 112.4 | 42.8 | 16.6 | 12.8 |
| Mistral | prompted | GPT-OSS-20B | 115.2 | 47.8 | 17.4 | 15.8 |
| Mistral | prompted | GPT-OSS-120B | 122.2 | 49.2 | 17.4 | 16.2 |
| DeepSeek | blind | Llama | 137.2 | 55.6 | 25.4 | 20.6 |
| DeepSeek | blind | GPT-OSS-20B | 149.4 | 61.6 | 27.4 | 19.8 |
| DeepSeek | blind | GPT-OSS-120B | 158.6 | 64.6 | 31.6 | 22.8 |
| DeepSeek | prompted | Llama | 157.2 | 64.2 | 30.4 | 20.6 |
| DeepSeek | prompted | GPT-OSS-20B | 165.2 | 68.2 | 32.4 | 23.0 |
| DeepSeek | prompted | GPT-OSS-120B | 176.4 | 73.4 | 33.0 | 24.2 |

## Findings (computed from the table above)

- **Qwen**: mean poisoned samples flagged across monitors fell from 53.1 at Gen 1 to 9.6 at Gen 12 (82% reduction) — consistent with **backdoor attenuation** (there are simply fewer poisoned samples left to catch, not that monitors got worse at catching them).
- **Mistral**: mean poisoned samples flagged across monitors fell from 111.3 at Gen 1 to 14.3 at Gen 12 (87% reduction) — consistent with **backdoor attenuation** (there are simply fewer poisoned samples left to catch, not that monitors got worse at catching them).
- **DeepSeek**: mean poisoned samples flagged across monitors fell from 157.3 at Gen 1 to 21.8 at Gen 12 (86% reduction) — consistent with **backdoor attenuation** (there are simply fewer poisoned samples left to catch, not that monitors got worse at catching them).
- **Behavioral diffusion**: see `diffusion_by_generation.png` / `artifacts/benchmark_scores/` — insecure-code rate on *untriggered* prompts rises from near baseline at Gen 1 and plateaus rather than returning to baseline, i.e. the conditional backdoor's effect diffuses into general behavior instead of simply disappearing alongside the explicit trigger.
- **Prompted vs. blind monitors**: within every (model, generation), the `prompted` rows are consistently higher than the matching `blind` row for the same monitor — telling a monitor that backdoors may be present (without revealing the trigger) measurably increases how many poisoned samples it catches. It never drives the flagged count to fully match the underlying poisoned rate, though, and the untriggered/diffuse poisoned rate keeps climbing regardless of protocol — matching the **irreversible drift** finding that review protocols help but do not eliminate degradation once diffusion has occurred.
