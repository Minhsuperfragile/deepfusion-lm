#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  run_prepare_sft_data.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Config ---
PATH_TO_DATA_STORE="$SCRIPT_DIR/data/tokenized-sft"
CONTEXT_LENGTH=1024
TEST_SPLIT_PCT=0.005
NUM_WORKERS=4
DATASET_SPLIT_SEED=42

echo "=== DeepfusionLM Preparing SFT Data ==="
echo "Output path  : $PATH_TO_DATA_STORE"
echo "Context len  : $CONTEXT_LENGTH tokens"
echo "Test split   : $(echo "$TEST_SPLIT_PCT * 100" | bc)%"
echo "Num workers  : $NUM_WORKERS"
echo ""

python "$SCRIPT_DIR/prepare_sft_data.py" \
    --path_to_data_store "$PATH_TO_DATA_STORE" \
    --context_length     $CONTEXT_LENGTH        \
    --test_split_pct     $TEST_SPLIT_PCT        \
    --num_workers        $NUM_WORKERS           \
    --dataset_split_seed $DATASET_SPLIT_SEED

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Done! Dataset saved to: $PATH_TO_DATA_STORE"
else
    echo ""
    echo "Script failed with exit code $EXIT_CODE" >&2
    exit $EXIT_CODE
fi
