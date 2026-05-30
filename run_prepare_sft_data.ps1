# ============================================================
#  run_prepare_sft_data.ps1
# ============================================================

# --- Config ---
$PATH_TO_DATA_STORE  = ".\data\tokenized-sft"
$CONTEXT_LENGTH      = 1024
$TEST_SPLIT_PCT      = 0.005
$NUM_WORKERS         = 4
$DATASET_SPLIT_SEED  = 42

Write-Host "=== DeepfusionLM Preparing SFT Data ===" -ForegroundColor Cyan
Write-Host "Output path  : $PATH_TO_DATA_STORE"
Write-Host "Context len  : $CONTEXT_LENGTH tokens"
Write-Host "Test split   : $($TEST_SPLIT_PCT * 100)%"
Write-Host "Num workers  : $NUM_WORKERS"
Write-Host ""

python prepare_sft_data.py `
    --path_to_data_store $PATH_TO_DATA_STORE `
    --context_length $CONTEXT_LENGTH `
    --test_split_pct $TEST_SPLIT_PCT `
    --num_workers $NUM_WORKERS `
    --dataset_split_seed $DATASET_SPLIT_SEED

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Done! Dataset saved to: $PATH_TO_DATA_STORE" -ForegroundColor Green
}
else {
    Write-Host ""
    Write-Host "Script failed with exit code $LASTEXITCODE" -ForegroundColor Red
}