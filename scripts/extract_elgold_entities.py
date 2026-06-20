"""Extract per-domain entity sets from ELGOLD, mapped to Wikidata Q-IDs.

This is a standalone analysis tool: it summarizes which gold entities (and how
many NIL mentions) appear in each ELGOLD domain. It is NOT required to run the
per-domain evaluation in evaluation/eval_accuracy.py — that script derives the
domain straight from the filename via utils/elgold_categories.py. Use this
script when you need entity-level statistics per domain (e.g. to report
coverage, or to scope a domain-specific candidate index).

ELGOLD data/ contains flat .txt files named <category_id>_<doc_num>.txt, see
utils/elgold_categories.py for the category_id -> domain name mapping.

Each file contains inline annotations `{{mention | class | target}}`; an empty
target denotes a NIL mention (no Wikipedia entity).

Setup (one time):
    pip install wikimapper
    wikimapper download enwiki-latest --dir wikimapper_data
    wikimapper create enwiki-latest --dumpdir wikimapper_data \\
        --target wikimapper_data/index_enwiki-latest.db

Usage:
    python scripts/extract_elgold_entities.py data/benchmarks \\
        data/elgold_entities_per_domain.json \\
        --wikimapper_db data/wikimapper/index_enwiki-latest.db
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from wikimapper import WikiMapper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.elgold_categories import category_from_filename

ENTITY_RE = re.compile(r"\{\{\s*([^|{}]+?)\s*\|\s*([^|{}]+?)\s*\|\s*([^{}]*?)\s*\}\}")


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
