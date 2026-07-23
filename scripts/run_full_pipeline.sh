#!/usr/bin/env bash
# Run the full generation-to-generation transfer pipeline for every model x
# protocol combination in the given config, chaining generations 0..N.
#
# Usage:
#   bash scripts/run_full_pipeline.sh configs/default.yaml

set -euo pipefail

CONFIG="${1:-configs/default.yaml}"

MODELS=$(python -c "import yaml,sys; print('\n'.join(m['name'] for m in yaml.safe_load(open(sys.argv[1]))['models']))" "$CONFIG")
NUM_GEN=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['transfer']['num_generations'])" "$CONFIG")
PROTOCOLS="none blind prompted"

for MODEL in $MODELS; do
  for PROTOCOL in $PROTOCOLS; do
    echo "=== $MODEL | protocol=$PROTOCOL ==="
    for GEN in $(seq 0 "$NUM_GEN"); do
      python scripts/run_single_generation.py \
        --config "$CONFIG" \
        --model "$MODEL" \
        --generation "$GEN" \
        --protocol "$PROTOCOL"
    done
    python -m src.eval.attack_success_rate --model "$MODEL" --protocol "$PROTOCOL"
  done
done

echo "Done. Inspect artifacts/manifest.jsonl for the full run history."
