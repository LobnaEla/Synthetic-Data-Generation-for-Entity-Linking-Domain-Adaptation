"""Convert synthetic EL data into the reranking training format.

Input (one line per mention): {paragraph, mention, start, end, entity_id, ...}.
Candidates for a mention = the distinct entities co-occurring in the SAME
paragraph (gold always included). The synthetic generator puts ~4 entities per
paragraph, so candidate lists are shorter than ZELDA's K=10 — accepted by
design. Prompt format is identical to filter_zelda_rerank.py (imported), so the
resulting model trains and evaluates exactly like the ZELDA rerank specialists.

Titles come from the ELGOLD per-domain mapping; descriptions from ZELDA's
entity_descriptions.jsonl. No wikimapper needed (entities are already Q-IDs).

Output: {prompt, ground_truth_id, domain, mention}.

Usage:
    python convert_synthetic_rerank.py "final dataset/dataset_Historic_blogs.jsonl" \\
        Historic_blogs ProjectDSAI2026_13/elgold_entities_per_domain.json \\
        zelda/other/entity_descriptions.jsonl --output synth_rerank/Historic_blogs.jsonl
"""

from filter_zelda_rerank import DESC_MAX_CHARS, build_prompt

import argparse
import json
from collections import OrderedDict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Convert synthetic EL data to reranking format.")
    p.add_argument("synthetic_jsonl", type=Path)
    p.add_argument("domain", help="ELGOLD domain name (e.g. Historic_blogs).")
    p.add_argument("elgold_json", type=Path)
    p.add_argument("descriptions_jsonl", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--context_chars", type=int, default=384)
    return p.parse_args()


def load_descriptions(path):
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            t = o.get("wikipedia_title")
            if t:
                out[t] = o.get("description", "")
    return out


def qid_to_title_map(elgold_json, domain):
    info = json.loads(elgold_json.read_text(encoding="utf-8"))[domain]
    m = {}
    for title, qid in info["title_to_qid"].items():
        m.setdefault(qid, title)  # first title wins
    return m


def describe(title, descriptions):
    for cand in (title, title.replace("_", " "), title.replace(" ", "_")):
        if cand in descriptions:
            return descriptions[cand][:DESC_MAX_CHARS]
    return ""


def main():
    args = parse_args()
    descriptions = load_descriptions(args.descriptions_jsonl)
    qid_title = qid_to_title_map(args.elgold_json, args.domain)

    # Group records by paragraph, preserving entity order of appearance.
    paragraphs = OrderedDict()
    for line in args.synthetic_jsonl.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        paragraphs.setdefault(r["paragraph"], []).append(r)

    n_out = 0
    n_no_title = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fout:
        for paragraph, records in paragraphs.items():
            # Distinct candidate entities in order of first appearance.
            cands = OrderedDict()
            for r in sorted(records, key=lambda x: x["start"]):
                qid = r["entity_id"]
                if qid in cands:
                    continue
                title = qid_title.get(qid)
                if not title:
                    n_no_title += 1
                    continue
                cands[qid] = {"qid": qid, "title": title,
                              "description": describe(title, descriptions)}
            cand_list = list(cands.values())
            if not cand_list:
                continue

            for r in records:
                prompt = build_prompt(
                    paragraph, r["start"], r["end"], args.context_chars, cand_list
                )
                fout.write(json.dumps({
                    "prompt": prompt,
                    "ground_truth_id": r["entity_id"],
                    "domain": [args.domain],
                    "mention": r["mention"],
                }, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"{args.domain}: {n_out} examples written, {len(paragraphs)} paragraphs, "
          f"avg {n_out/max(len(paragraphs),1):.1f} mentions/para, {n_no_title} skipped (no title)")
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
