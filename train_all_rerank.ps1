# Train one reranking specialist per ELGOLD domain, smallest dataset first.
#
# Idempotent and resumable: a domain whose final_adapter already exists is
# skipped; a domain with checkpoints but no final_adapter is resumed. Safe to
# re-run after a machine shutdown — it picks up where it left off.
#
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File ProjectDSAI2026_13\train_all_rerank.ps1

$env:CC = "C:\Users\dietl\AppData\Roaming\Python\Python313\site-packages\triton\runtime\tcc\tcc.exe"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# Smallest to largest by training-set size.
$domains = @(
    "Historic_blogs",
    "Scientific_abstracts",
    "Automotive_blogs",
    "Amazon_product_reviews",
    "Job_offers",
    "Movie_reviews",
    "News"
)

foreach ($d in $domains) {
    $out = "output_${d}_rerank"
    $train = "splits\rerank\$d\train.jsonl"
    $val = "splits\rerank\$d\val.jsonl"

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

Write-Host "All domains processed." -ForegroundColor Green
