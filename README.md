# backdoor-inheritance

Model organism for studying whether a backdoor / misalignment introduced in one
model generation survives into the models it helps train, and whether simple
AI-control-style review (a weaker "monitor" model reading the fine-tuning
data/code before it's used) breaks the chain.

This repo implements the pipeline described in the project brief and slide deck:

1. **Model organism construction** (`src/model_organism/`) — instantiate a
   trigger-conditioned backdoor (year-in-context → secure vs. insecure code),
   following the Sleeper Agents (Hubinger et al., 2024) recipe.
2. **Generation-to-generation transfer** (`src/training/`) — the current
   generation model produces fine-tuning data (or, for ICL organisms, is
   sampled in-context) to train/prompt its successor. Repeat for N generations.
3. **Monitors** (`src/monitors/`) — a weaker reviewer model screens the
   fine-tuning data/code under two protocols:
   - **Blind**: monitor has no idea backdoors might be present.
   - **Prompted**: monitor is told backdoors may exist, but not the trigger.
4. **Evaluation harness** (`src/eval/`) — with the trigger known to the
   evaluator (but not the monitor), measure attack success rate / poisoned
   sample count across generations and across monitor protocols.
5. **Artifact store** (`src/utils/artifact_store.py`) — a single, append-only
   place where every run writes what it generated. This is the "one place" to
   keep dropping in new data as you collect it — see below.

## Repo layout

```
backdoor-inheritance/
├── configs/
│   └── default.yaml          # models, trigger config, generation count, protocols
├── src/
│   ├── model_organism/
│   │   ├── triggers.py       # trigger definitions + secure/insecure code templates
│   │   └── generate_organism_data.py   # builds gen-0 poisoned fine-tuning set
│   ├── training/
│   │   ├── finetune.py       # LoRA fine-tune wrapper (HF transformers + peft)
│   │   └── generation_pipeline.py      # drives gen_k -> gen_{k+1} loop
│   ├── monitors/
│   │   ├── blind_monitor.py
│   │   ├── prompted_monitor.py
│   │   └── monitor_utils.py
│   ├── eval/
│   │   ├── attack_success_rate.py
│   │   └── benchmark_harness.py
│   └── utils/
│       ├── artifact_store.py # <-- central artifact read/write API
│       └── logging_utils.py
├── artifacts/                # <-- ALL generated data lands here (gitignored contents, kept dir)
│   ├── manifest.jsonl        # append-only index of every artifact ever written
│   └── README.md
├── scripts/
│   ├── run_full_pipeline.sh
│   └── run_single_generation.py
└── tests/
    └── test_triggers.py
```

## The artifact convention (read this first)

Every script in this repo writes through `src/utils/artifact_store.py` instead
of writing files ad hoc. That gives you one growing, append-only home for
everything the pipeline produces, no matter which model/generation/protocol
produced it:

```python
from src.utils.artifact_store import ArtifactStore

store = ArtifactStore()  # defaults to ./artifacts

store.save(
    kind="poisoned_finetune_data",   # what this artifact is
    model="qwen3.5-9b",
    generation=0,
    protocol=None,                    # or "blind" / "prompted" once a monitor is involved
    data=[{"prompt": "...", "completion": "..."}, ...],
    meta={"trigger_year": 2024, "num_poisoned": 42},
)
```

This:
- writes the payload to `artifacts/<kind>/<model>/gen_<n>/<timestamp>_<uuid>.jsonl`
- appends one line to `artifacts/manifest.jsonl` recording where it went and
  its metadata, so you can query "give me every artifact for Mistral gen 4
  under the prompted protocol" without opening every file.

To pull everything back out (e.g. for the results plot in the slide deck):

```python
records = store.query(kind="attack_success_rate", model="deepseek-v3")
```

As you run more experiments over time, just keep calling `store.save(...)` —
new runs never overwrite old ones, and the manifest keeps a full history.

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run_single_generation.py --config configs/default.yaml --generation 0
```

Then chain generations:

```bash
bash scripts/run_full_pipeline.sh configs/default.yaml
```

