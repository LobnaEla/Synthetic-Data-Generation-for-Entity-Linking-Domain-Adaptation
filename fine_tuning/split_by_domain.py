"""Fan out a filtered ZELDA JSONL into one file per ELGOLD domain.

Non-destructive: reads the filtered JSONL and writes NEW per-domain files to
output_dir. The original multi-domain file is left untouched, so both the
multi-domain model and the per-domain specialist models can be trained.

Each example's `domain` field is the list of ELGOLD domains its gold entity
belongs to (an entity can appear in several domains, e.g. "United States"
shows up in News, Automotive blogs, Scientific abstracts...). An example is
therefore copied into EVERY domain file it belongs to: per-domain training
sets overlap, and the sum of per-domain counts exceeds the total.

Important: the split is by ENTITY membership, not by text genre. All contexts
are Wikipedia prose (ZELDA). A "News specialist" thus learns the *entities*
that occur in ELGOLD News, not the News writing style.

Usage:
    python split_by_domain.py zelda_filtered_train.jsonl by_domain/generative
    # then make train/val splits for each domain file, e.g.:
    #   python make_splits.py by_domain/generative/News.jsonl splits/generative/News
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Split a filtered ZELDA JSONL into one file per ELGOLD domain."
    )
    p.add_argument("input_jsonl", type=Path)
    p.add_argument("output_dir", type=Path)
    return p.parse_args()


def domains_of(example):
    d = example.get("domain", [])
    return d if isinstance(d, list) else [d]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    handles = {}
    counts = Counter()
    n_total = 0
    try:
        with args.input_jsonl.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                n_total += 1
                for domain in domains_of(json.loads(line)):
                    if domain not in handles:
                        handles[domain] = (
                            args.output_dir / f"{domain}.jsonl"
                        ).open("w", encoding="utf-8")
                    handles[domain].write(line + "\n")
                    counts[domain] += 1
    finally:
        for h in handles.values():
            h.close()

    print(f"Total examples read: {n_total}")
    print(f"{'Domain':30s} {'examples':>9s}")
    for domain, c in counts.most_common():
        print(f"{domain:30s} {c:9d}")
    print("\n(sum > total: an entity can belong to several domains)")
    print(f"Written to: {args.output_dir}")


if __name__ == "__main__":
    main()
