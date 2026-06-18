import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidates.bm25_retriever import BM25Retriever
from candidates.dense_retriever import DenseRetriever
from utils.elgold_parser import load_benchmark
from utils.metrics import recall_at_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", required=True, choices=["bm25", "dense"])
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--bm25-index", default="candidates/index/bm25.pkl")
    parser.add_argument("--dense-index", default="candidates/index/")
    parser.add_argument("--benchmarks", default="data/benchmarks")
    parser.add_argument("--out", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_path = args.out or f"candidates/precomputed/{args.retriever}_k{args.k}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Load benchmark
    df = load_benchmark(args.benchmarks)
    print(f"{len(df)} mentions loaded")

    # Resume from existing file
    results = {}
    if args.resume and Path(out_path).exists():
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)
        print(f"[RESUME] {len(results)} mentions already computed")

    # Load retriever
    if args.retriever == "bm25":
        retriever = BM25Retriever.load(args.bm25_index)
    else:
        retriever = DenseRetriever.load(args.dense_index)

    # Retrieve candidates (skip already done)
    remaining = [
        (_, row) for _, row in df.iterrows()
        if f"{row['file']}::{row['mention']}::{row['gold']}" not in results
    ]
    print(f"Retrieving top-{args.k} candidates for {len(remaining)} mentions ...")

    for i, (_, row) in enumerate(tqdm(remaining, desc=args.retriever)):
        key = f"{row['file']}::{row['mention']}::{row['gold']}"
        results[key] = retriever.retrieve(row["mention"], k=args.k)

        # Checkpoint every 500 mentions
        if (i + 1) % 500 == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)

    # Final save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    # Recall
    recall_count = sum(
        recall_at_k(row["gold"], results[f"{row['file']}::{row['mention']}::{row['gold']}"])
        for _, row in df.iterrows()
    )
    recall = recall_count / len(df)
    print(f"Recall@{args.k} : {recall:.2%}")
    print(f"Saved: {out_path} ({len(results)} mentions)")


if __name__ == "__main__":
    main()