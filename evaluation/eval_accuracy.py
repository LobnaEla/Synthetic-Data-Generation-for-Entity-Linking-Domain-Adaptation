import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidates.dense_retriever import DenseRetriever
from candidates.bm25_retriever import BM25Retriever
from candidates.reranker import Reranker
from utils.elgold_parser import load_benchmark
from utils.llm import generate_batch, load_model
from utils.metrics import compute_metrics, is_correct, print_metrics, recall_at_k
from utils.prompt_builder import build_mcq_prompt
from utils.response_parser import parse_mcq_response

TITLES_PATH = "data/wikidata_qid_titles.csv"
INDEX_PATH = "candidates/index/bm25.pkl"
BENCHMARKS = "data/benchmarks"
DENSE_INDEX_PATH = "candidates/index/"
PRECOMPUTED_DIR = "candidates/precomputed"
RESULTS_DIR = "evaluation/results"
TITLE_COL = "title"
K = 10
RERANKER_MODEL = "Qwen/Qwen3-Reranker-4B"

# Candidate loading
 
def load_precomputed(pattern: str) -> dict | None:
    """Auto-detect precomputed candidates matching a glob pattern."""
    matches = sorted(Path(PRECOMPUTED_DIR).glob(pattern))
    if not matches:
        return None
    path = matches[-1]
    print(f"[CAND] Found pre-computed: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
 
 
def get_candidates_for_row(row, precomputed: dict | None, retriever=None, k: int = 100):
    """Get candidates for a single row: precomputed or live retrieval."""
    key = f"{row['file']}::{row['mention']}::{row['gold']}"
    if precomputed and key in precomputed:
        return precomputed[key]
    if retriever:
        return retriever.retrieve(row["mention"], k=k)
    return []
 
 
def load_candidates(df: pd.DataFrame, mode: str, args) -> tuple:
    """Load/compute candidates based on mode. Returns (df, k_retrieval, info_str)."""
 
    if mode == "fusion":
        # Load both BM25 and Dense, merge
        bm25_pre = load_precomputed("bm25_k*.json")
        dense_pre = load_precomputed("dense_k*.json")
 
        bm25_retriever = None
        dense_retriever = None
        if not bm25_pre:
            print("[CAND] No BM25 precomputed, loading live retriever")
            bm25_retriever = BM25Retriever.load(args.index)
        if not dense_pre:
            print("[CAND] No Dense precomputed, loading live retriever")
            dense_retriever = DenseRetriever.load(args.dense_index)
 
        def fuse(row):
            bm25_cands = get_candidates_for_row(row, bm25_pre, bm25_retriever, k=100)
            dense_cands = get_candidates_for_row(row, dense_pre, dense_retriever, k=64)
            # Union, preserving order (BM25 first, then dense additions)
            seen = set()
            merged = []
            for c in bm25_cands + dense_cands:
                if c not in seen:
                    seen.add(c)
                    merged.append(c)
            return merged
 
        print("[CAND] Fusing BM25 + Dense candidates ...")
        df["candidates"] = df.apply(fuse, axis=1)
        k_max = max(len(c) for c in df["candidates"] if len(c) > 0)
        return df, k_max, "fusion (BM25 + Dense)"
 
    elif mode == "zero_shot_bm25":
        pre = load_precomputed("bm25_k*.json")
        retriever = None if pre else BM25Retriever.load(args.index)
        df["candidates"] = df.apply(
            lambda row: get_candidates_for_row(row, pre, retriever, k=100), axis=1
        )
        k_max = max(len(c) for c in df["candidates"] if len(c) > 0)
        return df, k_max, "BM25"
 
    elif mode == "zero_shot_dense":
        pre = load_precomputed("dense_k*.json")
        retriever = None if pre else DenseRetriever.load(args.dense_index)
        df["candidates"] = df.apply(
            lambda row: get_candidates_for_row(row, pre, retriever, k=64), axis=1
        )
        k_max = max(len(c) for c in df["candidates"] if len(c) > 0)
        return df, k_max, "Dense"
 
    else:
        raise ValueError(f"Unknown mode: {mode}")
 
 
# Evaluation loop
 
def run_evaluation(
    df: pd.DataFrame,
    tokenizer,
    model,
    reranker: Reranker | None = None,
    k: int = 10,
    batch_size: int = 8,
    max_new_tokens: int = 2048,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    results = []
    records = df.to_dict("records")
 
    # Resume
    done_keys = set()
    if resume and checkpoint_path and checkpoint_path.exists():
        df_done = pd.read_csv(checkpoint_path)
        results = df_done.to_dict("records")
        done_keys = {
            f"{r['file']}::{r['mention']}::{r['gold']}" for r in results
        }
        print(f"[RESUME] {len(done_keys)} examples already processed")
 
    remaining = [
        rec for rec in records
        if f"{rec['file']}::{rec['mention']}::{rec['gold']}" not in done_keys
    ]
    print(f"[EVAL] {len(remaining)} examples to process ({len(done_keys)} already done)")
 
    # candidates_reranked is prepared in main().
    # If a row does not have it, fall back to all retrieved candidates without top-k truncation.
    for rec in remaining:
        if "candidates_reranked" not in rec:
            rec["candidates_reranked"] = list(rec.get("candidates", []))
 
    # LLM evaluation
    for i in tqdm(range(0, len(remaining), batch_size), desc="eval"):
        batch = remaining[i : i + batch_size]
 
        prompts = [
            build_mcq_prompt(
                rec["context"], rec["mention"], rec["entity_type"],
                rec["candidates_reranked"]
            )
            for rec in batch
        ]
 
        try:
            raw_outputs = generate_batch(prompts, tokenizer, model, max_new_tokens)
        except Exception as e:
            for rec in batch:
                results.append({
                    "file": rec["file"], "mention": rec["mention"],
                    "entity_type": rec["entity_type"], "gold": rec["gold"],
                    "candidates_retrieval": rec.get("candidates", []),
                    "candidates_reranked": rec.get("candidates_reranked", []),
                    "gold_in_retrieval": recall_at_k(rec["gold"], rec.get("candidates", [])),
                    "gold_in_reranked": recall_at_k(rec["gold"], rec.get("candidates_reranked", [])),
                    "predicted": "", "correct": False, "raw_output": "", "error": str(e),
                })
            continue
 
        for rec, raw in zip(batch, raw_outputs):
            try:
                parsed = parse_mcq_response(raw, candidates=rec["candidates_reranked"])
                pred, error = parsed["predicted"], ""
            except Exception as e:
                pred, error = "", str(e)
 
            results.append({
                "file":                rec["file"],
                "mention":             rec["mention"],
                "entity_type":         rec["entity_type"],
                "gold":                rec["gold"],
                "candidates_retrieval": rec.get("candidates", []),
                "candidates_reranked": rec.get("candidates_reranked", []),
                "gold_in_retrieval":   recall_at_k(rec["gold"], rec.get("candidates", [])),
                "gold_in_reranked":    recall_at_k(rec["gold"], rec.get("candidates_reranked", [])),
                "predicted":           pred,
                "correct":             is_correct(pred, rec["gold"]) if not error else False,
                "raw_output":          raw,
                "error":               error,
            })
 
        if checkpoint_path:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
 
    return pd.DataFrame(results)
 
 
# Main 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["zero_shot_bm25", "zero_shot_dense", "fusion"])
    parser.add_argument("--model", required=True, help="HuggingFace LLM for selection")
    parser.add_argument("--reranker", default=RERANKER_MODEL,
                        help=f"Reranker model (default: {RERANKER_MODEL})")
    parser.add_argument("--no-reranker", action="store_true",
                        help="Skip reranking, pass all retrieved candidates to LLM (no top-k truncation)")
    parser.add_argument("--k", type=int, default=K,
                        help="Candidates after reranking; ignored when --no-reranker is used (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--index", default=INDEX_PATH)
    parser.add_argument("--dense-index", default=DENSE_INDEX_PATH)
    parser.add_argument("--benchmarks", default=BENCHMARKS)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
 
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
 
    # 1. Load benchmark
    print("[DATA] Loading ELGOLD ...")
    df = load_benchmark(args.benchmarks, args.max_examples)
    print(f"[DATA] {len(df)} examples  |  files: {df['file'].nunique()}")
 
    # 2. Load candidates (precomputed or live)
    df, k_retrieval, retrieval_info = load_candidates(df, args.mode, args)
 
    df["gold_in_candidates"] = df.apply(
        lambda row: recall_at_k(row["gold"], row["candidates"]), axis=1
    )
    recall_retrieval = df["gold_in_candidates"].mean()
    print(f"[CAND] Retrieval: {retrieval_info}")
    print(f"[CAND] Recall@{k_retrieval} (retrieval) : {recall_retrieval:.2%}")
 
    # 3. Rerank BEFORE loading the LLM.
    # Important on P100 16GB: do not keep Qwen3-4B LLM and Qwen3-Reranker-4B
    # loaded at the same time. The reranking loop stays inside candidates/reranker.py.
    used_reranker = not args.no_reranker
    if used_reranker:
        reranker = Reranker(args.reranker)
        df = reranker.rerank_dataframe(df, k=args.k, unload_after=True)
    else:
        print("[RERANK] Skipped: passing all retrieved candidates to LLM (no top-k truncation).")
        df["candidates_reranked"] = df["candidates"].apply(lambda c: list(c))
 
    # 4. Load LLM AFTER reranker has been unloaded.
    tokenizer, model = load_model(args.model)
 
    # 5. Run evaluation
    slug = args.model.replace("/", "_")
    checkpoint_path = Path(args.results_dir) / f"{args.mode}_{slug}_checkpoint.csv"
 
    if used_reranker:
        print(f"\n[EVAL] Pipeline: {retrieval_info} → Reranker → LLM ({args.model}) → top-{args.k}")
    else:
        print(f"\n[EVAL] Pipeline: {retrieval_info} → LLM ({args.model}) → all retrieved candidates")
 
    df_results = run_evaluation(
        df, tokenizer, model,
        reranker=None,
        k=args.k,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
    )
 
    # 6. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(args.results_dir) / f"{args.mode}_{slug}_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n[SAVE] Results → {csv_path}")
 
    if checkpoint_path.exists():
        checkpoint_path.unlink()
 
    # 7. Metrics
    metrics = compute_metrics(df_results)
    print_metrics(metrics, model_name=f"{args.mode} / {args.model}")
 
    # Retrieval recall vs reranked recall
    metrics["recall_retrieval"] = recall_retrieval
    metrics["k_retrieval"] = k_retrieval
    if "gold_in_reranked" in df_results.columns:
        recall_reranked = df_results["gold_in_reranked"].mean()
        metrics["recall_reranked"] = recall_reranked
        print(f"Recall@{k_retrieval} (retrieval) : {recall_retrieval:.2%}")
        if used_reranker:
            metrics["k_reranked"] = args.k
            print(f"Recall@{args.k} (reranked)  : {recall_reranked:.2%}")
        else:
            metrics["k_reranked"] = k_retrieval
            print(f"Recall@{k_retrieval} (LLM candidates, no reranker) : {recall_reranked:.2%}")
    metrics["llm_loss"] = round(
        (metrics.get("recall_reranked", recall_retrieval)) - metrics["accuracy"], 4
    )
    print(f"LLM loss : -{abs(metrics['llm_loss']):.2%}")
 
    json_path = Path(args.results_dir) / f"{args.mode}_{slug}_{timestamp}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics → {json_path}")
 
 
if __name__ == "__main__":
    main()