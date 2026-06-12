import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidates.bm25_retriever import BM25Retriever
from utils.elgold_parser import load_benchmark
from utils.llm import generate_batch, load_model, majority_vote
from utils.metrics import compute_metrics, is_correct, print_metrics, recall_at_k
from utils.prompt_builder import build_prompt_mcq, build_prompt_zero_shot
from utils.response_parser import parse_response
from utils.wiki_descriptions import fetch_descriptions_batch

TITLES_PATH = "data/wikidata_qid_titles.csv"
INDEX_PATH = "candidates/index/bm25.pkl"
BENCHMARKS = "data/benchmarks"
RESULTS_DIR = "evaluation/results"
TITLE_COL = "title"
K = 10


# BM25
def load_or_build_index(index_path: str, titles_path: str) -> BM25Retriever:
    if Path(index_path).exists():
        print(f"[BM25] Loading index from {index_path}")
        return BM25Retriever.load(index_path)

    print(f"[BM25] Building index from {titles_path} ...")
    titles_df = pd.read_csv(titles_path)
    if TITLE_COL not in titles_df.columns:
        raise ValueError(
            f"Column '{TITLE_COL}' not found in {titles_path}. "
            f"Available: {titles_df.columns.tolist()}"
        )
    titles = titles_df[TITLE_COL].dropna().tolist()
    print(f"[BM25] {len(titles):,} titles loaded")
    retriever = BM25Retriever.from_entity_list(titles)
    retriever.save(index_path)
    return retriever


# evaluation loop
def run_evaluation(
    df: pd.DataFrame,
    mode: str,
    tokenizer,
    model,
    n_samples: int = 3,
    temperature: float = 0.6,
    max_new_tokens: int = 2048,
) -> pd.DataFrame:
    results = []
    records = df.to_dict("records")

    # Pre-fetch all descriptions (fills cache, avoids per-mention API calls)
    if mode != "zero_shot":
        all_candidates = set(c for rec in records for c in rec.get("candidates", []))
        print(f"[DESC] Pre-fetching descriptions for {len(all_candidates):,} unique candidates...")
        fetch_descriptions_batch(list(all_candidates))

    for rec in tqdm(records, desc=f"{mode}"):
        if mode == "zero_shot":
            prompt = build_prompt_zero_shot(rec["context"], rec["mention"], rec["entity_type"])
        else:
            descs = fetch_descriptions_batch(rec["candidates"])
            prompt = build_prompt_mcq(
                rec["context"], rec["mention"], rec["entity_type"],
                rec["candidates"], descriptions=descs
            )

        try:
            raw_outputs = generate_batch(
                prompt, tokenizer, model,
                n_samples=1 if mode == "zero_shot" else n_samples,
                max_new_tokens=50 if mode == "zero_shot" else max_new_tokens,
                temperature=temperature,
                do_sample=mode != "zero_shot",
            )

            if mode == "zero_shot":
                parsed = parse_response(raw_outputs[0], mode="zero_shot")
                pred = parsed["predicted"]
                raw_output = raw_outputs[0]
                n_agree = 1
            else:
                voted = majority_vote(raw_outputs, rec["candidates"])
                pred = voted["predicted"]
                raw_output = str(voted["raw_outputs"])
                n_agree = voted["n_agree"]

            error = ""

        except Exception as e:
            pred, raw_output, n_agree, error = "", "", 0, str(e)

        results.append({
            "file":               rec["file"],
            "mention":            rec["mention"],
            "entity_type":        rec["entity_type"],
            "gold":               rec["gold"],
            "candidates":         rec.get("candidates", []),
            "gold_in_candidates": rec.get("gold_in_candidates", None),
            "predicted":          pred,
            "correct":            is_correct(pred, rec["gold"]) if not error else False,
            "raw_output":         raw_output,
            "n_agree":            n_agree,
            "error":              error,
        })

    return pd.DataFrame(results)


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",           required=True, choices=["zero_shot", "mcq_bm25"])
    parser.add_argument("--model",          required=True, help="HuggingFace model id")
    parser.add_argument("--max-examples",   type=int, default=None)
    parser.add_argument("--n-samples",      type=int, default=3)
    parser.add_argument("--temperature",    type=float, default=0.6)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--titles",         default=TITLES_PATH)
    parser.add_argument("--index",          default=INDEX_PATH)
    parser.add_argument("--benchmarks",     default=BENCHMARKS)
    parser.add_argument("--results-dir",    default=RESULTS_DIR)
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark
    print("[DATA] Loading ELGOLD ...")
    df = load_benchmark(args.benchmarks, args.max_examples)
    print(f"[DATA] {len(df)} examples  |  files: {df['file'].nunique()}")

    # 2. BM25 retrieval
    recall = None
    if args.mode == "mcq_bm25":
        retriever = load_or_build_index(args.index, args.titles)
        print(f"[BM25] Retrieving top-{K} candidates for each mention ...")
        df["candidates"] = df["mention"].apply(lambda m: retriever.retrieve(m, k=K))
        df["gold_in_candidates"] = df.apply(
            lambda row: recall_at_k(row["gold"], row["candidates"]), axis=1
        )
        recall = df["gold_in_candidates"].mean()
        print(f"[BM25] Recall@{K} : {recall:.2%}")
    else:
        df["candidates"]         = [[] for _ in range(len(df))]
        df["gold_in_candidates"] = [None] * len(df)

    # 3. Load model
    tokenizer, model = load_model(args.model)

    # 4. Run evaluation
    print(f"\n[EVAL] Running {args.mode} with {args.model} "
          f"(n_samples={args.n_samples}, temp={args.temperature}) ...")
    df_results = run_evaluation(
        df, args.mode, tokenizer, model,
        n_samples=args.n_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # 5. Save results
    slug = args.model.replace("/", "_")
    csv_path = Path(args.results_dir) / f"{args.mode}_{slug}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n[SAVE] Results → {csv_path}")

    # 6. Metrics
    metrics = compute_metrics(df_results)
    print_metrics(metrics, model_name=f"{args.mode} / {args.model}")

    if recall is not None:
        metrics["recall_at_k_bm25"] = recall
        metrics["llm_loss"] = round(recall - metrics["accuracy"], 4)
        print(f"BM25 ceiling : {recall:.2%}")
        print(f"LLM loss     : -{(recall - metrics['accuracy']):.2%}")

    json_path = Path(args.results_dir) / f"{args.mode}_{slug}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics  → {json_path}")


if __name__ == "__main__":
    main()