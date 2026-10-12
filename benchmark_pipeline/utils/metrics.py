import re
import pandas as pd


def normalize_entity(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip().replace("_", " ").lower()


def is_correct(pred: str, gold: str) -> bool:
    return normalize_entity(pred) == normalize_entity(gold)


def recall_at_k(gold: str, candidates: list[str]) -> bool:
    gold_norm = normalize_entity(gold)
    return any(normalize_entity(c) == gold_norm for c in candidates)


def compute_metrics(df: pd.DataFrame) -> dict:
    df_valid = df[df["error"] == ""] if "error" in df.columns else df
    n_total = len(df_valid)
    n_correct = int(df_valid["correct"].sum()) if n_total > 0 else 0
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "n_errors": len(df) - n_total,
    }

    if "candidates" in df_valid.columns:
        recall = df_valid.apply(
            lambda row: recall_at_k(row["gold"], row["candidates"])
            if isinstance(row["candidates"], list)
            else False,
            axis=1,
        ).mean()
        metrics["recall_at_k"] = recall

    return metrics


def print_metrics(metrics: dict, model_name: str = "") -> None:
    label = f" [{model_name}]" if model_name else ""
    print(f"\n{'='*50}{label}")
    print(f"  Accuracy   : {metrics['accuracy']:.2%}  ({metrics['n_correct']}/{metrics['n_total']})")
    if "recall_at_k" in metrics:
        print(f"  Recall@K   : {metrics['recall_at_k']:.2%}  (gold in candidates)")
    if metrics["n_errors"] > 0:
        print(f"  Errors     : {metrics['n_errors']} skipped")
    print("=" * 50)