# Train one reranking specialist per domain on SYNTHETIC data, smallest first.
#
# Mirror of train_all_rerank.ps1 but reads splits\synth_rerank and writes to
# output_<Domain>_synth_rerank. Idempotent and resumable after a shutdown.
#
#   powershell -ExecutionPolicy Bypass -File ProjectDSAI2026_13\train_all_synth_rerank.ps1

$env:CC = "C:\Users\dietl\AppData\Roaming\Python\Python313\site-packages\triton\runtime\tcc\tcc.exe"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# Smallest to largest by training-set size.
$domains = @(
    "Automotive_blogs",
    "Historic_blogs",
    "Job_offers",
    "News",
    "Movie_reviews",
    "Amazon_product_reviews",
    "Scientific_abstracts"
)

foreach ($d in $domains) {
    $out = "output_${d}_synth_rerank"
    $train = "splits\synth_rerank\$d\train.jsonl"
    $val = "splits\synth_rerank\$d\val.jsonl"

    if (Test-Path "$out\final_adapter\adapter_model.safetensors") {
        Write-Host "[$d] final_adapter present -> skip" -ForegroundColor Green
        continue
    }

    $args = @(
        "-u", "ProjectDSAI2026_13\finetune_unsloth.py",
        $train, $out, "--val_data", $val,
        "--epochs", "1", "--save_steps", "250", "--eval_steps", "250",
        "--report_to", "none"
    )

    $hasCkpt = (Test-Path $out) -and (Get-ChildItem $out -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue)
    if ($hasCkpt) {
        $args += "--resume"
        Write-Host "[$d] resuming from checkpoint..." -ForegroundColor Yellow
    } else {
        Write-Host "[$d] training from scratch..." -ForegroundColor Cyan
    }

    $start = Get-Date
    python @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[$d] FAILED (exit $LASTEXITCODE) -- continuing to next domain" -ForegroundColor Red
    } else {
        $mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
        Write-Host "[$d] done in $mins min" -ForegroundColor Green
    }
}

Write-Host "All synthetic domains processed." -ForegroundColor Green
