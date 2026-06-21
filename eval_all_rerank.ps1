# Evaluate every reranking specialist on its own ELGOLD domain.
#
# Idempotent: a domain whose report already exists is skipped; a domain whose
# adapter is missing is skipped (not yet trained). Safe to re-run.
#
# Run from the project root, after training:
#   powershell -ExecutionPolicy Bypass -File ProjectDSAI2026_13\eval_all_rerank.ps1

$env:CC = "C:\Users\dietl\AppData\Roaming\Python\Python313\site-packages\triton\runtime\tcc\tcc.exe"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

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
    $adapter = "output_${d}_rerank\final_adapter"
    $prefix = "eval_${d}_rerank"

    if (-not (Test-Path "$adapter\adapter_model.safetensors")) {
        Write-Host "[$d] no adapter -> skip" -ForegroundColor Yellow
        continue
    }
    if (Test-Path "${prefix}_report.json") {
        Write-Host "[$d] already evaluated -> skip" -ForegroundColor Green
        continue
    }

    Write-Host "[$d] evaluating..." -ForegroundColor Cyan
    $args = @(
        "-u", "ProjectDSAI2026_13\evaluate_elgold.py",
        "data", $adapter,
        "ProjectDSAI2026_13\elgold_entities_per_domain.json",
        "--mode", "rerank", "--domain", $d,
        "--alias_pickle", "zelda\other\zelda_mention_entities_counter.pickle",
        "--descriptions_jsonl", "zelda\other\entity_descriptions.jsonl",
        "--wikimapper_db", "wikimapper_data\index_enwiki-latest.db",
        "--train_jsonl", "splits\rerank\$d\train.jsonl",
        "--output_prefix", $prefix
    )
    python @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[$d] eval FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
    }
}

Write-Host "All evaluations processed." -ForegroundColor Green
