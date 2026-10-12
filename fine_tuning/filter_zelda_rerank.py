"""Filter ZELDA train into a *reranking* dataset for finetune_unsloth.py.

For each ZELDA mention whose gold Q-ID is in the ELGOLD entity set, build a
prompt listing top-K candidate entities retrieved from ZELDA's alias table
(`zelda_mention_entities_counter.pickle`), each with a short description from
`entity_descriptions.jsonl`. Add an explicit "NONE" option for the case where
the gold is not among the retrieved candidates.

Output JSONL fields: prompt, ground_truth_id ("NONE" or a Q-ID), domain, mention.

Usage:
    python filter_zelda_rerank.py path/to/zelda_train.jsonl \\
        elgold_entities_per_domain.json \\
        path/to/zelda_mention_entities_counter.pickle \\
        path/to/entity_descriptions.jsonl \\
        --wikimapper_db wikimapper_data/index_enwiki-latest.db \\
        --output zelda_rerank_train.jsonl
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

from wikimapper import WikiMapper

NONE_TOKEN = "NONE"
DESC_MAX_CHARS = 150

PROMPT_TEMPLATE = (
    "Identify the Wikidata entity for the mention marked with **double "
    "asterisks** in the context.\n\n"
    "Context:\n{context_before}**{mention}**{context_after}\n\n"
    "Candidates:\n{candidates}\n\n"
    "Which candidate does the mention refer to? Answer with only the Q-ID "
    "(e.g. Q42) or \"NONE\"."
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Filter ZELDA into a reranking dataset for entity linking."
    )
    p.add_argument("zelda_jsonl", type=Path)
    p.add_argument("elgold_json", type=Path)
    p.add_argument("alias_pickle", type=Path,
                   help="zelda_mention_entities_counter.pickle")
    p.add_argument("descriptions_jsonl", type=Path,
                   help="entity_descriptions.jsonl")
    p.add_argument("--wikimapper_db", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("zelda_rerank_train.jsonl"))
    p.add_argument("--k", type=int, default=10, help="Number of candidates per mention.")
    p.add_argument("--context_chars", type=int, default=384)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def load_qid_to_domains(path: Path):
    summary = json.loads(path.read_text(encoding="utf-8"))
    qid_to_domains = {}
    for domain, info in summary.items():
        for qid in info["entities_qids"]:
            qid_to_domains.setdefault(qid, []).append(domain)
    return qid_to_domains


def load_descriptions(path: Path):
    """Return dict mapping wikipedia_title -> short description string."""
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            title = obj.get("wikipedia_title")
            desc = obj.get("description", "")
            if title:
                out[title] = desc
    return out


class TitleQidCache:
    """Memoized title -> Q-ID resolver to amortize wikimapper SQLite lookups."""
    def __init__(self, mapper: WikiMapper):
        self.mapper = mapper
        self.cache = {}

    def get(self, title: str):
        if title in self.cache:
            return self.cache[title]
        qid = self.mapper.title_to_id(title.replace(" ", "_")) \
              or self.mapper.title_to_id(title)
        self.cache[title] = qid
        return qid


def iter_mentions(doc):
    spans = doc.get("index", [])
    titles = doc.get("mention_gold_titles") or doc.get("wikipedia_titles") or []
    wiki_ids = doc.get("mention_gold_ids") or doc.get("wikipedia_ids") or []
    n = min(len(spans), len(titles))
    for i in range(n):
        start, end = spans[i]
        yield start, end, titles[i], (wiki_ids[i] if i < len(wiki_ids) else None)


def build_candidates(alias_table, mention_surface, k, qid_cache, descriptions):
    """Return list of dicts: [{qid, title, description}], top-K by alias count, deduped by Q-ID."""
    counter = alias_table.get(mention_surface)
    if not counter:
        return []
    seen_qids = set()
    out = []
    for title, _ in sorted(counter.items(), key=lambda x: -x[1]):
        qid = qid_cache.get(title)
        if not qid or qid in seen_qids:
            continue
        seen_qids.add(qid)
        out.append({
            "qid": qid,
            "title": title,
            "description": descriptions.get(title, "")[:DESC_MAX_CHARS],
        })
        if len(out) >= k:
            break
    return out


def format_candidates(cands):
    lines = []
    for c in cands:
        desc = f": {c['description']}" if c["description"] else ""
        lines.append(f"  {c['qid']} — {c['title']}{desc}")
    lines.append(f"  {NONE_TOKEN} — none of the candidates above")
    return "\n".join(lines)


def build_prompt(text, start, end, ctx, cands):
    a = max(0, start - ctx)
    b = min(len(text), end + ctx)
    return PROMPT_TEMPLATE.format(
        context_before=text[a:start],
        mention=text[start:end],
        context_after=text[end:b],
        candidates=format_candidates(cands),
    )


def main():
    args = parse_args()
    print("Loading ELGOLD entity targets...")
    qid_to_domains = load_qid_to_domains(args.elgold_json)
    target_qids = set(qid_to_domains)

    print("Loading alias table...")
    with args.alias_pickle.open("rb") as f:
        alias_table = pickle.load(f)

    print("Loading entity descriptions...")
    descriptions = load_descriptions(args.descriptions_jsonl)

    print("Opening wikimapper DB...")
    mapper = WikiMapper(str(args.wikimapper_db))
    qid_cache = TitleQidCache(mapper)

    stats = Counter()
    domain_counter = Counter()
    completion_dist = Counter()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.zelda_jsonl.open("r", encoding="utf-8") as fin, \
         args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            if args.limit is not None and stats["docs"] >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            stats["docs"] += 1
            text = doc.get("text", "")
            if not text:
                continue
            for start, end, gold_title, gold_wid in iter_mentions(doc):
                stats["mentions"] += 1
                gold_qid = qid_cache.get(gold_title) if gold_title else None
                if gold_qid is None and gold_wid is not None:
                    try:
                        gold_qid = mapper.wikipedia_id_to_id(int(gold_wid))
                    except (ValueError, TypeError):
                        gold_qid = None
                if gold_qid is None:
                    stats["no_gold_qid"] += 1
                    continue
                if gold_qid not in target_qids:
                    stats["off_target"] += 1
                    continue

                mention_surface = text[start:end]
                cands = build_candidates(
                    alias_table, mention_surface, args.k, qid_cache, descriptions
                )
                if not cands:
                    stats["no_candidates"] += 1
                    continue

                cand_qids = {c["qid"] for c in cands}
                completion = gold_qid if gold_qid in cand_qids else NONE_TOKEN

                fout.write(json.dumps({
                    "prompt": build_prompt(text, start, end, args.context_chars, cands),
                    "ground_truth_id": completion,
                    "domain": qid_to_domains[gold_qid],  # always a list: stable pyarrow type
                    "mention": mention_surface,
                }, ensure_ascii=False) + "\n")
                stats["kept"] += 1
                completion_dist[completion if completion == NONE_TOKEN else "Q-ID"] += 1
                for d in qid_to_domains[gold_qid]:
                    domain_counter[d] += 1

    print(f"\nDocs processed:      {stats['docs']}")
    print(f"Mentions seen:       {stats['mentions']}")
    print(f"  no gold Q-ID:      {stats['no_gold_qid']}")
    print(f"  off-target:        {stats['off_target']}")
    print(f"  no candidates:     {stats['no_candidates']}")
    print(f"  kept:              {stats['kept']}")
    print(f"\nCompletion distribution among kept:")
    for k, v in completion_dist.most_common():
        pct = 100 * v / max(stats["kept"], 1)
        print(f"  {k:10s} {v:6d}  ({pct:.1f}%)")
    print("\nKept per ELGOLD domain:")
    for d, c in domain_counter.most_common():
        print(f"  {d:30s} {c:6d}")
    print(f"\nQ-ID cache size: {len(qid_cache.cache)}")
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
