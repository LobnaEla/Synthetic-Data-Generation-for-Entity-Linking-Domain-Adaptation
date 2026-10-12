"""Report ELGOLD-entity coverage of a filtered ZELDA training set, per domain.

For each ELGOLD domain, measures which fraction of its gold entities (Q-IDs)
appears at least once in the filtered training data. Low coverage on a domain
means evaluation there is closer to zero-shot than to fine-tuning — critical
context when comparing branch (2) vs branch (3).

Usage:
    python coverage_report.py zelda_filtered_train.jsonl \\
        ProjectDSAI2026_13/elgold_entities_per_domain.json \\
        --output coverage_report.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="ELGOLD coverage report for a filtered training set.")
    p.add_argument("train_jsonl", type=Path, help="Filtered ZELDA JSONL (train).")
    p.add_argument("elgold_json", type=Path, help="elgold_entities_per_domain.json.")
    p.add_argument("--output", type=Path, default=Path("coverage_report.json"))
    return p.parse_args()


def main():
    args = parse_args()
    elgold = json.loads(args.elgold_json.read_text(encoding="utf-8"))

    train_qid_counts = Counter()
    with args.train_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid = json.loads(line)["ground_truth_id"]
            if qid != "NONE":
                train_qid_counts[qid] += 1

    report = {}
    print(f"{'Domain':30s} {'elgold':>7s} {'covered':>8s} {'coverage':>9s} "
          f"{'median#':>8s} {'zero-shot':>10s}")
    for domain, info in elgold.items():
        gold = set(info["entities_qids"])
        covered = {q for q in gold if train_qid_counts[q] > 0}
        missing = sorted(gold - covered)
        counts_covered = sorted(train_qid_counts[q] for q in covered)
        median = counts_covered[len(counts_covered) // 2] if counts_covered else 0
        report[domain] = {
            "n_elgold_entities": len(gold),
            "n_covered": len(covered),
            "coverage": round(len(covered) / len(gold), 4) if gold else None,
            "median_train_mentions_per_covered_entity": median,
            "missing_qids": missing,
        }
        print(f"{domain:30s} {len(gold):7d} {len(covered):8d} "
              f"{100 * len(covered) / len(gold):8.1f}% {median:8d} {len(missing):10d}")

    all_gold = {q for info in elgold.values() for q in info["entities_qids"]}
    all_covered = {q for q in all_gold if train_qid_counts[q] > 0}
    report["_global"] = {
        "n_elgold_entities": len(all_gold),
        "n_covered": len(all_covered),
        "coverage": round(len(all_covered) / len(all_gold), 4),
        "n_train_examples": sum(train_qid_counts.values()),
        "n_distinct_train_qids": len(train_qid_counts),
    }
    print(f"\n{'GLOBAL':30s} {len(all_gold):7d} {len(all_covered):8d} "
          f"{100 * len(all_covered) / len(all_gold):8.1f}%")
    print(f"Train examples: {report['_global']['n_train_examples']}, "
          f"distinct Q-IDs in train: {report['_global']['n_distinct_train_qids']}")

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
