#!/usr/bin/env bash
set -euo pipefail

echo "=== DeepfusionLM SFT Fine-tuning ==="

# ─── GPU Selection ────────────────────────────────────────────────────────────
# Đặt ID của GPU muốn dùng. Ví dụ:
#   "0"     → chỉ dùng GPU 0
#   "1"     → chỉ dùng GPU 1
#   "0,1"   → multi-GPU (GPU 0 và 1)
GPU_IDS="0"

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOKENIZER_PATH="$SCRIPT_DIR/DeepfusionLM_tokenizer"
CHECKPOINT="$SCRIPT_DIR/checkpoint-pretrained-with-history-qa/model.safetensors"
DATA="$SCRIPT_DIR/data/tokenized-sft"
WORKING_DIR="$SCRIPT_DIR"
EXPERIMENT_NAME="sft_run"

# ─── Training hyperparams ────────────────────────────────────────────────────
PER_GPU_BATCH=8
GRAD_ACCUM=4          # effective batch = 32
NUM_EPOCHS=5          # số epochs (float ok, vd: 0.5 = nửa epoch)
WARMUP_STEPS=200
LR="2e-5"
WEIGHT_DECAY="0.01"
NUM_WORKERS=4

# ─── Logging / checkpointing ─────────────────────────────────────────────────
LOG_EVERY=10
EVAL_EVERY=500
CKPT_EVERY=500

EFFECTIVE_BATCH=$((PER_GPU_BATCH * GRAD_ACCUM))

echo "GPU(s)     : $GPU_IDS  (CUDA_VISIBLE_DEVICES)"
echo "Tokenizer  : $TOKENIZER_PATH"
echo "Checkpoint : $CHECKPOINT"
echo "Data       : $DATA"
echo "Epochs     : $NUM_EPOCHS  |  batch=$PER_GPU_BATCH (accum=$GRAD_ACCUM, effective=$EFFECTIVE_BATCH)"
echo "LR         : $LR  |  warmup=$WARMUP_STEPS steps"
echo ""

# ─── Set GPU và chạy ─────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

python "$SCRIPT_DIR/finetune_sft.py" \
    --experiment_name               "$EXPERIMENT_NAME"   \
    --working_directory             "$WORKING_DIR"       \
    --tokenizer_path                "$TOKENIZER_PATH"    \
    --path_to_pretrained_checkpoint "$CHECKPOINT"        \
    --path_to_prepped_data          "$DATA"              \
    --log_wandb                                          \
    --wandb_project                 "deepfusion-sft"     \
    --per_gpu_batch_size            $PER_GPU_BATCH       \
    --gradient_accumulation_steps   $GRAD_ACCUM          \
    --num_epochs                    $NUM_EPOCHS          \
    --num_warmup_steps              $WARMUP_STEPS        \
    --learning_rate                 $LR                  \
    --weight_decay                  $WEIGHT_DECAY        \
    --num_workers                   $NUM_WORKERS         \
    --logging_steps                 $LOG_EVERY           \
    --evaluation_interval           $EVAL_EVERY          \
    --checkpoint_interval           $CKPT_EVERY

EXIT_CODE=$?

# ─── Unset GPU env sau khi chạy xong ─────────────────────────────────────────
unset CUDA_VISIBLE_DEVICES

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Done! Model saved to: $WORKING_DIR/$EXPERIMENT_NAME/final_model"
else
    echo "Script failed with exit code: $EXIT_CODE" >&2
    exit $EXIT_CODE
fi
