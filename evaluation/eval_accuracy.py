import argparse
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidates.bm25_retriever import BM25Retriever
from utils.prompt_builder import build_prompt_mcq, build_prompt_zero_shot
from utils.response_parser import parse_response
from utils.metrics import compute_metrics, is_correct, print_metrics, recall_at_k

TITLES_PATH = "data/wikidata_qid_titles.csv"
INDEX_PATH = "candidates/index/bm25.pkl"
BENCHMARKS = "data/benchmarks"
RESULTS_DIR = "evaluation/results"
TITLE_COL = "title" 
K = 10


# ELGOLD parsing
ANNOT_RE = re.compile(r"\{\{([^|{}]+)\|([^|{}]+)\|([^{}]+)\}\}")

def parse_elgold_file(path: Path, context_window: int = 150) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    clean = ANNOT_RE.sub(lambda m: m.group(1).strip(), raw)
    examples, cursor, last_end = [], 0, 0
    for match in ANNOT_RE.finditer(raw):
        mention = match.group(1).strip()
        ent_type = match.group(2).strip()
        gold = match.group(3).strip()
        cursor += len(raw[last_end : match.start()])
        end = cursor + len(mention)
        snippet = clean[max(0, cursor - context_window) : min(len(clean), end + context_window)]
        examples.append({
            "file": path.name, "mention": mention,
            "entity_type": ent_type, "gold": gold, "context": snippet,
        })
        cursor, last_end = end, match.end()
    return examples


def load_benchmark(benchmarks_dir: str, max_examples: int | None) -> pd.DataFrame:
    rows = []
    for f in glob.glob(f"{benchmarks_dir}/*.txt"):
        rows.extend(parse_elgold_file(Path(f)))
    df = pd.DataFrame(rows).drop_duplicates()
    if max_examples:
        df = df.head(max_examples)
    return df


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


# LLM
def load_model(model_name: str):
    print(f"[LLM] Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    ).eval()
    device = next(model.parameters()).device
    print(f"[LLM] Loaded on {device}")
    return tokenizer, model


def generate_batch(
    prompts: list[str],
    tokenizer,
    model,
    max_new_tokens: int = 20,
) -> list[str]:
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=1024,
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in outputs
    ]


# evaluation loop
def run_evaluation(
    df: pd.DataFrame,
    mode: str,
    tokenizer,
    model,
    batch_size: int,
) -> pd.DataFrame:
    llm_mode = "zero_shot" if mode == "zero_shot" else "mcq"
    results = []
    records = df.to_dict("records")

    for i in tqdm(range(0, len(records), batch_size), desc=f"{mode}"):
        batch = records[i : i + batch_size]
        prompts = [
            build_prompt_zero_shot(r["context"], r["mention"], r["entity_type"])
            if mode == "zero_shot"
            else build_prompt_mcq(r["context"], r["mention"], r["entity_type"], r["candidates"])
            for r in batch
        ]

        try:
            raws = generate_batch(prompts, tokenizer, model)
        except Exception as e:
            for r in batch:
                results.append({**r, "predicted": "", "correct": False,
                                 "raw_output": "", "error": str(e)})
            continue

        for rec, raw in zip(batch, raws):
            try:
                parsed = parse_response(raw, mode=llm_mode, candidates=rec.get("candidates"))
                pred, error = parsed["predicted"], ""
            except Exception as e:
                pred, error = "", str(e)

            results.append({
                "file": rec["file"],
                "mention": rec["mention"],
                "entity_type": rec["entity_type"],
                "gold": rec["gold"],
                "candidates": rec.get("candidates", []),
                "gold_in_candidates": rec.get("gold_in_candidates", None),
                "predicted": pred,
                "correct": is_correct(pred, rec["gold"]) if not error else False,
                "raw_output": raw,
                "error": error,
            })

    return pd.DataFrame(results)


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["zero_shot", "mcq_bm25"])
    parser.add_argument("--model", required=True, help="HuggingFace model id")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--titles", default=TITLES_PATH)
    parser.add_argument("--index", default=INDEX_PATH)
    parser.add_argument("--benchmarks", default=BENCHMARKS)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark
    print("[DATA] Loading ELGOLD ...")
    df = load_benchmark(args.benchmarks, args.max_examples)
    print(f"[DATA] {len(df)} examples  |  files: {df['file'].nunique()}")

    # 2. BM25 retrieval (skip for zero_shot)
    recall = None
    if args.mode == "mcq_bm25":
        retriever = load_or_build_index(args.index, args.titles)

        print(f"[BM25] Retrieving top-{K} candidates for each mention ...")
        df["candidates"] = df["mention"].apply(lambda m: retriever.retrieve(m, k=K))
        df["gold_in_candidates"] = df.apply(
            lambda row: recall_at_k(row["gold"], row["candidates"]), axis=1
        )
        recall = df["gold_in_candidates"].mean()
        print(f"[BM25] Recall@{K} : {recall:.2%}  ← accuracy ceiling for MCQ")
    else:
        df["candidates"]         = [[] for _ in range(len(df))]
        df["gold_in_candidates"] = [None] * len(df)

    # 3. Load model
    tokenizer, model = load_model(args.model)

    # 4. Run evaluation
    print(f"\n[EVAL] Running {args.mode} with {args.model} ...")
    df_results = run_evaluation(df, args.mode, tokenizer, model, args.batch_size)

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
        print(f"LLM loss : -{(recall - metrics['accuracy']):.2%}")

    json_path = Path(args.results_dir) / f"{args.mode}_{slug}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics  → {json_path}")


if __name__ == "__main__":
    main()