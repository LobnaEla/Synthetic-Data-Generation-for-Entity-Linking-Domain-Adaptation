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
from utils.llm import generate_batch_safe, load_model
from utils.metrics import compute_metrics, is_correct, print_metrics, recall_at_k
from utils.prompt_builder import build_mcq_prompt
from utils.qid_lookup import load_title_to_qid
from utils.response_parser import parse_mcq_response

TITLES_PATH = "data/wikidata_qid_titles.csv"
INDEX_PATH = "candidates/index/bm25.pkl"
BENCHMARKS = "data/benchmarks"
DENSE_INDEX_PATH = "candidates/index/"
PRECOMPUTED_DIR = "candidates/precomputed"
RERANKED_DIR = "candidates/reranked"
RESULTS_DIR = "evaluation/results"
TITLE_COL = "title"
K = 10
RERANKER_MODEL = "Qwen/Qwen3-Reranker-4B"


def row_key(row) -> str:
    return f"{row['file']}::{row['mention']}::{row['gold']}"


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
    key = row_key(row)
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


# Reranked candidates persistence (stage="rerank" writes, stage="llm" reads)

def reranked_path(reranked_dir: str, mode: str, domain: str, k: int) -> Path:
    return Path(reranked_dir) / f"{mode}_{domain}_k{k}.json"


def save_reranked(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {row_key(row): row["candidates_reranked"] for _, row in df.iterrows()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"[RERANK] Saved reranked candidates → {path} ({len(data)} mentions)")


def load_reranked(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No pre-computed reranked candidates found at {path}. "
            f"Run with --stage rerank (or --stage full) first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Evaluation loop

def run_evaluation(
    df: pd.DataFrame,
    tokenizer,
    model,
    title_to_qid: dict[str, str],
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

    # candidates_reranked is prepared by the caller (rerank stage or precomputed load).
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
                rec["candidates_reranked"], title_to_qid
            )
            for rec in batch
        ]

        try:
            # generate_batch_safe halves the batch on CUDA OOM instead of dropping
            # every prompt in it; a single prompt that still OOMs alone is reported
            # as an error only for that example.
            gen_outputs = generate_batch_safe(prompts, tokenizer, model, max_new_tokens)
        except Exception as e:
            gen_outputs = [("", str(e))] * len(batch)

        for rec, (raw, gen_error) in zip(batch, gen_outputs):
            if gen_error:
                pred, error = "", gen_error
            else:
                try:
                    parsed = parse_mcq_response(
                        raw, candidates=rec["candidates_reranked"], title_to_qid=title_to_qid
                    )
                    pred, error = parsed["predicted"], ""
                except Exception as e:
                    pred, error = "", str(e)

            results.append({
                "file":                rec["file"],
                "domain":              rec.get("domain", ""),
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


# Per-domain stage helpers

def run_retrieval_and_rerank(df_domain, domain, args, reranker: Reranker | None):
    """Stage 'rerank' / first half of stage 'full': retrieval -> optional reranking.

    Returns (df_domain, recall_retrieval, k_retrieval, retrieval_info).
    """
    df_domain, k_retrieval, retrieval_info = load_candidates(df_domain, args.mode, args)

    df_domain["gold_in_candidates"] = df_domain.apply(
        lambda row: recall_at_k(row["gold"], row["candidates"]), axis=1
    )
    recall_retrieval = df_domain["gold_in_candidates"].mean()
    print(f"[{domain}] Retrieval: {retrieval_info}")
    print(f"[{domain}] Recall@{k_retrieval} (retrieval) : {recall_retrieval:.2%}")

    if reranker is not None:
        df_domain = reranker.rerank_dataframe(df_domain, k=args.k, unload_after=False)
    else:
        print(f"[{domain}] Reranking skipped: passing all retrieved candidates through.")
        df_domain["candidates_reranked"] = df_domain["candidates"].apply(lambda c: list(c))

    return df_domain, recall_retrieval, k_retrieval, retrieval_info


def attach_precomputed_reranked(df_domain, mode: str, domain: str, k: int, reranked_dir: str):
    """Stage 'llm': load pre-computed reranked candidates for this domain from disk.

    Retrieval candidates are not available in this stage, so 'candidates' is set
    to the same reranked list (gold_in_retrieval == gold_in_reranked here).
    """
    path = reranked_path(reranked_dir, mode, domain, k)
    reranked = load_reranked(path)
    print(f"[{domain}] Loaded pre-computed reranked candidates ← {path}")

    df_domain = df_domain.copy()
    df_domain["candidates_reranked"] = df_domain.apply(
        lambda row: reranked.get(row_key(row), []), axis=1
    )
    df_domain["candidates"] = df_domain["candidates_reranked"]
    return df_domain


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["zero_shot_bm25", "zero_shot_dense", "fusion"])
    parser.add_argument("--stage", default="full", choices=["full", "rerank", "llm"],
                        help="full: retrieval+rerank+LLM. rerank: retrieval+rerank only, "
                             "saves reranked candidates and exits (no LLM loaded). "
                             "llm: skip retrieval/rerank, load pre-computed reranked "
                             "candidates and run the LLM stage only.")
    parser.add_argument("--domains", default="all",
                        help="Comma-separated list of ELGOLD domains to evaluate "
                             "(e.g. News,Job_offers), or 'all' (default). Evaluation "
                             "is run, and results saved, separately per domain.")
    parser.add_argument("--model", default=None,
                        help="HuggingFace LLM for selection (required unless --stage rerank)")
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
    parser.add_argument("--titles", default=TITLES_PATH,
                        help="Title->QID lookup CSV, shown to the LLM as candidate ids "
                             "instead of plain indices.")
    parser.add_argument("--benchmarks", default=BENCHMARKS)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--reranked-dir", default=RERANKED_DIR,
                        help="Where reranked candidates are saved (--stage rerank/full) "
                             "or read from (--stage llm).")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.stage != "rerank" and not args.model:
        parser.error("--model is required unless --stage rerank")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark (each row already tagged with its ELGOLD 'domain')
    print("[DATA] Loading ELGOLD ...")
    df = load_benchmark(args.benchmarks, args.max_examples)
    print(f"[DATA] {len(df)} examples  |  files: {df['file'].nunique()}  |  "
          f"domains: {sorted(df['domain'].unique())}")

    available_domains = sorted(df["domain"].unique())
    if args.domains == "all":
        domains = available_domains
    else:
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]
        unknown = set(domains) - set(available_domains)
        if unknown:
            parser.error(f"Unknown domain(s): {sorted(unknown)}. Available: {available_domains}")

    used_reranker = not args.no_reranker
    slug = args.model.replace("/", "_") if args.model else None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    domain_frames: dict[str, pd.DataFrame] = {}
    domain_retrieval_stats: dict[str, dict] = {}

    # 2-3. Retrieval + reranking, run for every selected domain before touching the LLM.
    # Important on P100 16GB: do not keep Qwen3-4B LLM and Qwen3-Reranker-4B
    # loaded at the same time. The reranking loop stays inside candidates/reranker.py.
    if args.stage in ("full", "rerank"):
        reranker = Reranker(args.reranker) if used_reranker else None

        for domain in domains:
            # --resume: a domain whose reranked candidates were already saved in a
            # previous run is loaded back instead of re-running retrieval + reranking.
            # Only applies when k is known upfront (i.e. the reranker truncates to --k);
            # without a reranker, k depends on live retrieval and is always recomputed.
            if used_reranker and args.resume:
                existing = reranked_path(args.reranked_dir, args.mode, domain, args.k)
                if existing.exists():
                    print(f"[RESUME][{domain}] Found existing {existing}, skipping retrieval+rerank.")
                    df_domain = attach_precomputed_reranked(
                        df[df["domain"] == domain].reset_index(drop=True),
                        args.mode, domain, args.k, args.reranked_dir,
                    )
                    domain_frames[domain] = df_domain
                    domain_retrieval_stats[domain] = {
                        "recall_retrieval": df_domain.apply(
                            lambda row: recall_at_k(row["gold"], row["candidates"]), axis=1
                        ).mean(),
                        "k_retrieval": args.k,
                        "retrieval_info": "pre-computed reranked candidates (resumed)",
                    }
                    continue

            df_domain = df[df["domain"] == domain].reset_index(drop=True)
            df_domain, recall_retrieval, k_retrieval, retrieval_info = run_retrieval_and_rerank(
                df_domain, domain, args, reranker
            )

            out_k = args.k if used_reranker else k_retrieval
            save_reranked(df_domain, reranked_path(args.reranked_dir, args.mode, domain, out_k))

            domain_frames[domain] = df_domain
            domain_retrieval_stats[domain] = {
                "recall_retrieval": recall_retrieval,
                "k_retrieval": k_retrieval,
                "retrieval_info": retrieval_info,
            }

        if reranker is not None:
            reranker.unload()

    if args.stage == "rerank":
        print("\n[STAGE] rerank complete — reranked candidates saved, no LLM was loaded.")
        for domain, stats in domain_retrieval_stats.items():
            print(f"  [{domain}] Recall@{stats['k_retrieval']} (retrieval): "
                  f"{stats['recall_retrieval']:.2%}")
        return

    # 4. Load LLM once, after the reranker has been unloaded (stage="full")
    #    or directly (stage="llm", no reranker is ever loaded).
    tokenizer, model = load_model(args.model)

    try:
        title_to_qid = load_title_to_qid(args.titles)
        print(f"[QID] Loaded {len(title_to_qid):,} title->QID mappings from {args.titles}")
    except FileNotFoundError:
        print(f"[QID] {args.titles} not found, candidates will use fallback ids (C0, C1, ...)")
        title_to_qid = {}

    all_metrics = {}

    for domain in domains:
        if args.stage == "full":
            df_domain = domain_frames[domain]
            retrieval_info = domain_retrieval_stats[domain]["retrieval_info"]
            recall_retrieval = domain_retrieval_stats[domain]["recall_retrieval"]
            k_retrieval = domain_retrieval_stats[domain]["k_retrieval"]
        else:  # stage == "llm"
            df_domain = df[df["domain"] == domain].reset_index(drop=True)
            df_domain = attach_precomputed_reranked(
                df_domain, args.mode, domain, args.k, args.reranked_dir
            )
            retrieval_info = "pre-computed reranked candidates"
            recall_retrieval = df_domain.apply(
                lambda row: recall_at_k(row["gold"], row["candidates"]), axis=1
            ).mean()
            k_retrieval = args.k

        checkpoint_path = Path(args.results_dir) / f"{args.mode}_{domain}_{slug}_checkpoint.csv"

        if used_reranker:
            print(f"\n[EVAL][{domain}] Pipeline: {retrieval_info} → Reranker → LLM ({args.model}) → top-{args.k}")
        else:
            print(f"\n[EVAL][{domain}] Pipeline: {retrieval_info} → LLM ({args.model}) → all retrieved candidates")

        df_results = run_evaluation(
            df_domain, tokenizer, model, title_to_qid,
            k=args.k,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
        )

        # 6. Save per-domain results
        csv_path = Path(args.results_dir) / f"{args.mode}_{domain}_{slug}_{timestamp}.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"[SAVE][{domain}] Results → {csv_path}")

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        # 7. Per-domain metrics
        metrics = compute_metrics(df_results)
        print_metrics(metrics, model_name=f"{args.mode} / {domain} / {args.model}")

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

        json_path = Path(args.results_dir) / f"{args.mode}_{domain}_{slug}_{timestamp}_metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SAVE][{domain}] Metrics → {json_path}")

        all_metrics[domain] = metrics

    # Summary across all evaluated domains
    summary_path = Path(args.results_dir) / f"{args.mode}_{slug}_{timestamp}_domains_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[SAVE] Per-domain summary → {summary_path}")

    print(f"\n{'='*60}\n Per-domain accuracy summary [{args.mode} / {args.model}]\n{'='*60}")
    for domain, metrics in all_metrics.items():
        print(f"  {domain:30s} accuracy={metrics['accuracy']:.2%}  "
              f"(n={metrics['n_total']}, errors={metrics['n_errors']})")


if __name__ == "__main__":
    main()
