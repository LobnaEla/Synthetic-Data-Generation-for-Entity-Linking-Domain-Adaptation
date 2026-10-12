"""Extract per-domain entity sets from ELGOLD, mapped to Wikidata Q-IDs.

ELGOLD data/ contains flat .txt files named <category_id>_<doc_num>.txt where
category_id is one of {1, 2, 3, 4, 5, 6a..6e, 8}. Subcategories of category 6
(Scientific abstracts) are aggregated, following the elgold-toolset convention
(`parsed_file['category'][0]` -> "ignore subcategories").

Each file contains inline annotations `{{mention | class | target}}`; an empty
target denotes a NIL mention (no Wikipedia entity).

Setup (one time):
    pip install wikimapper
    wikimapper download enwiki-latest --dir wikimapper_data
    wikimapper create enwiki-latest --dumpdir wikimapper_data \\
        --target wikimapper_data/index_enwiki-latest.db

Usage:
    python extract_elgold_entities.py path/to/elgold/data \\
        elgold_entities_per_domain.json \\
        --wikimapper_db wikimapper_data/index_enwiki-latest.db
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from wikimapper import WikiMapper

ENTITY_RE = re.compile(r"\{\{\s*([^|{}]+?)\s*\|\s*([^|{}]+?)\s*\|\s*([^{}]*?)\s*\}\}")

CATEGORY_NAMES = {
    "1": "News",
    "2": "Job_offers",
    "3": "Movie_reviews",
    "4": "Automotive_blogs",
    "5": "Amazon_product_reviews",
    "6": "Scientific_abstracts",
    "8": "Historic_blogs",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract ELGOLD entity sets per domain, mapped to Wikidata Q-IDs."
    )
    p.add_argument("data_root", type=Path, help="Path to ELGOLD data/ (flat .txt files).")
    p.add_argument("output", type=Path, help="Output JSON summary file.")
    p.add_argument(
        "--wikimapper_db", type=Path, required=True,
        help="Path to the wikimapper SQLite DB.",
    )
    p.add_argument("--glob", default="*.txt", help="Glob for ELGOLD files (default: *.txt).")
    return p.parse_args()


def category_from_filename(name: str) -> str:
    """`6a_3.txt` -> 'Scientific_abstracts'. Aggregates sub-categories (6a..6e -> 6)."""
    prefix = name.split("_", 1)[0]
    numeric = "".join(c for c in prefix if c.isdigit())
    return CATEGORY_NAMES.get(numeric, f"category_{numeric or prefix}")


def parse_mentions(text: str):
    out = []
    for m in ENTITY_RE.finditer(text):
        out.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
    return out


def title_to_qid(mapper: WikiMapper, title: str):
    if not title:
        return None
    for candidate in (title.replace(" ", "_"), title):
        qid = mapper.title_to_id(candidate)
        if qid:
            return qid
    return None


def main():
    args = parse_args()
    mapper = WikiMapper(str(args.wikimapper_db))

    docs_per_cat = defaultdict(int)
    mentions_per_cat = defaultdict(list)
    nil_per_cat = defaultdict(int)

    for path in args.data_root.glob(args.glob):
        cat = category_from_filename(path.name)
        docs_per_cat[cat] += 1
        for mention, ne_class, target in parse_mentions(
            path.read_text(encoding="utf-8")
        ):
            if not target:
                nil_per_cat[cat] += 1
            else:
                mentions_per_cat[cat].append((mention, ne_class, target))

    summary = {}
    for cat in docs_per_cat:
        mentions = mentions_per_cat[cat]
        titles = sorted({t for _, _, t in mentions})
        title_qid = {}
        unmapped = []
        for t in titles:
            qid = title_to_qid(mapper, t)
            if qid:
                title_qid[t] = qid
            else:
                unmapped.append(t)
        summary[cat] = {
            "n_docs": docs_per_cat[cat],
            "n_mentions": len(mentions),
            "n_nil": nil_per_cat[cat],
            "n_unique_titles": len(titles),
            "n_mapped_qids": len(set(title_qid.values())),
            "n_unmapped_titles": len(unmapped),
            "entities_qids": sorted(set(title_qid.values())),
            "title_to_qid": title_qid,
            "unmapped_titles": unmapped,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    header = (
        f"{'Category':30s} {'docs':>5s} {'mentions':>8s} "
        f"{'unique':>7s} {'qids':>5s} {'unmapped':>9s} {'nil':>4s}"
    )
    print(header)
    for cat, s in summary.items():
        print(
            f"{cat:30s} {s['n_docs']:5d} {s['n_mentions']:8d} "
            f"{s['n_unique_titles']:7d} {s['n_mapped_qids']:5d} "
            f"{s['n_unmapped_titles']:9d} {s['n_nil']:4d}"
        )
    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
