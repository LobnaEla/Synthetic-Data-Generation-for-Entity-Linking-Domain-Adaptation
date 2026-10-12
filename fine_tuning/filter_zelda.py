"""Filter ZELDA train to mentions whose gold entity falls in the ELGOLD entity set.

Produces a JSONL with {prompt, ground_truth_id, domain, mention} suitable for
finetune_unsloth.py.

ZELDA JSONL schema (per flairNLP/zelda README): each line is a document with
fields including `text`, `index` (char offsets per mention), and either
`mention_gold_titles`/`mention_gold_ids` (gold) or `wikipedia_titles`/`wikipedia_ids`.
This script tries gold-* fields first, falls back to wikipedia_* if absent.
Wikipedia title -> Q-ID is resolved via wikimapper (with redirect handling).

Usage:
    python filter_zelda.py path/to/zelda_train.jsonl \\
        elgold_entities_per_domain.json \\
        --wikimapper_db wikimapper_data/index_enwiki-latest.db \\
        --output zelda_filtered_train.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from wikimapper import WikiMapper

PROMPT_TEMPLATE = (
    "Identify the Wikidata Q-ID of the entity referred to by the mention "
    "marked with **double asterisks** in the following text.\n\n"
    "Text:\n{context_before}**{mention}**{context_after}\n\n"
    "Answer with only the Q-ID (e.g. Q42)."
)


def parse_args():
    p = argparse.ArgumentParser(description="Filter ZELDA train by ELGOLD entity set.")
    p.add_argument("zelda_jsonl", type=Path, help="Path to ZELDA train JSONL.")
    p.add_argument("elgold_json", type=Path, help="Path to elgold_entities_per_domain.json.")
    p.add_argument("--wikimapper_db", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("zelda_filtered_train.jsonl"))
    p.add_argument("--context_chars", type=int, default=512,
                   help="Half-window of context (chars) on each side of the mention.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N ZELDA docs (debug).")
    return p.parse_args()


def load_qid_to_domains(path: Path):
    summary = json.loads(path.read_text(encoding="utf-8"))
    qid_to_domains = {}
    for domain, info in summary.items():
        for qid in info["entities_qids"]:
            qid_to_domains.setdefault(qid, []).append(domain)
    return qid_to_domains


def resolve_qid(mapper: WikiMapper, title, wiki_id):
    """Try title then numeric Wikipedia ID. Return Q-ID string or None."""
    if title:
        for candidate in (title.replace(" ", "_"), title):
            qid = mapper.title_to_id(candidate)
            if qid:
                return qid
    if wiki_id is not None:
        try:
            return mapper.wikipedia_id_to_id(int(wiki_id))
        except (ValueError, TypeError):
            return None
    return None


def iter_mentions(doc):
    """Yield (start, end, title, wiki_id) tuples for each gold mention in a ZELDA doc."""
    spans = doc.get("index", [])
    titles = doc.get("mention_gold_titles") or doc.get("wikipedia_titles") or []
    wiki_ids = doc.get("mention_gold_ids") or doc.get("wikipedia_ids") or []
    n = min(len(spans), len(titles))
    for i in range(n):
        start, end = spans[i]
        title = titles[i] if i < len(titles) else None
        wid = wiki_ids[i] if i < len(wiki_ids) else None
        yield start, end, title, wid


def build_prompt(text, start, end, ctx):
    a = max(0, start - ctx)
    b = min(len(text), end + ctx)
    return PROMPT_TEMPLATE.format(
        context_before=text[a:start],
        mention=text[start:end],
        context_after=text[end:b],
    )


def main():
    args = parse_args()
    qid_to_domains = load_qid_to_domains(args.elgold_json)
    target_qids = set(qid_to_domains)
    mapper = WikiMapper(str(args.wikimapper_db))

    n_docs = 0
    n_mentions = 0
    n_no_qid = 0
    n_off_target = 0
    n_kept = 0
    domain_counter = Counter()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.zelda_jsonl.open("r", encoding="utf-8") as fin, \
         args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            if args.limit is not None and n_docs >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            n_docs += 1
            text = doc.get("text", "")
            if not text:
                continue
            for start, end, title, wid in iter_mentions(doc):
                n_mentions += 1
                qid = resolve_qid(mapper, title, wid)
                if qid is None:
                    n_no_qid += 1
                    continue
                if qid not in target_qids:
                    n_off_target += 1
                    continue
                domains = qid_to_domains[qid]
                fout.write(json.dumps({
                    "prompt": build_prompt(text, start, end, args.context_chars),
                    "ground_truth_id": qid,
                    "domain": domains,  # always a list: pyarrow needs a stable column type
                    "mention": text[start:end],
                }, ensure_ascii=False) + "\n")
                n_kept += 1
                for d in domains:
                    domain_counter[d] += 1

    print(f"Docs processed: {n_docs}")
    print(f"Mentions seen:  {n_mentions}")
    print(f"  no Q-ID:      {n_no_qid}")
    print(f"  off-target:   {n_off_target}")
    print(f"  kept:         {n_kept}")
    print("\nKept mentions per ELGOLD domain (one mention may count in several):")
    for d, c in domain_counter.most_common():
        print(f"  {d:30s} {c:6d}")
    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
