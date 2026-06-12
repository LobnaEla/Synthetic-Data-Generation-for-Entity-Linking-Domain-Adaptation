"""Evaluate a fine-tuned EL model (LoRA adapter) on the ELGOLD benchmark.

Builds test prompts with the *exact same* templates as training (imported from
filter_zelda.py / filter_zelda_rerank.py), runs batched greedy inference, and
reports accuracy@1 per ELGOLD domain and globally.

Protocol decisions (must stay identical for branch (3)):
- NIL mentions (empty target) are EXCLUDED from scoring (counted separately).
- A prediction is correct iff the extracted Q-ID equals the gold Q-ID.
- Rerank mode: candidates come from ZELDA's alias table exactly as in training;
  a NONE prediction is always counted as incorrect end-to-end, but candidate
  recall and NONE-rate are reported for diagnosis.
- Greedy decoding (do_sample=False), max_new_tokens=16.

Usage (generative):
    python evaluate_elgold.py data output_generative/final_adapter \\
        ProjectDSAI2026_13/elgold_entities_per_domain.json \\
        --mode generative --output_prefix eval_generative

Usage (rerank):
    python evaluate_elgold.py data output_rerank/final_adapter \\
        ProjectDSAI2026_13/elgold_entities_per_domain.json \\
        --mode rerank \\
        --alias_pickle zelda/other/zelda_mention_entities_counter.pickle \\
        --descriptions_jsonl zelda/other/entity_descriptions.jsonl \\
        --wikimapper_db wikimapper_data/index_enwiki-latest.db \\
        --output_prefix eval_rerank
"""

# Unsloth must be imported before transformers/peft (it patches them).
from unsloth import FastLanguageModel

import argparse
import csv
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import torch

# Reuse training-time templates and helpers so eval prompts match exactly.
from extract_elgold_entities import ENTITY_RE, category_from_filename
from filter_zelda import PROMPT_TEMPLATE as GEN_TEMPLATE
from filter_zelda_rerank import (
    NONE_TOKEN,
    PROMPT_TEMPLATE as RERANK_TEMPLATE,
    TitleQidCache,
    build_candidates,
    format_candidates,
    load_descriptions,
)

QID_RE = re.compile(r"\bQ\d+\b")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a LoRA adapter on ELGOLD.")
    p.add_argument("elgold_dir", type=Path, help="ELGOLD data/ dir (flat .txt files).")
    p.add_argument("adapter", help="Path to the trained LoRA adapter directory.")
    p.add_argument("elgold_json", type=Path, help="elgold_entities_per_domain.json.")
    p.add_argument("--mode", choices=["generative", "rerank"], required=True)
    p.add_argument("--alias_pickle", type=Path, default=None, help="Rerank mode only.")
    p.add_argument("--descriptions_jsonl", type=Path, default=None, help="Rerank mode only.")
    p.add_argument("--wikimapper_db", type=Path, default=None,
                   help="Rerank mode only: same DB as training, so candidate "
                        "title->Q-ID resolution matches the training distribution.")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--context_chars", type=int, default=None,
                   help="Default: 512 (generative) / 384 (rerank), as in training.")
    p.add_argument("--max_seq_length", type=int, default=5120)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--train_jsonl", type=Path, default=None,
                   help="Optional: tag each mention seen/unseen vs training Q-IDs.")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only N mentions (debug).")
    p.add_argument("--output_prefix", default="eval_elgold")
    return p.parse_args()


def parse_elgold_doc(raw_text):
    """Strip {{mention|class|target}} markup.

    Returns (clean_text, mentions) where each mention is
    (start, end, surface, ne_class, target_title) with offsets in clean_text.
    """
    clean = []
    pos = 0
    mentions = []
    length = 0
    for m in ENTITY_RE.finditer(raw_text):
        before = raw_text[pos:m.start()]
        clean.append(before)
        length += len(before)
        surface = m.group(1).strip()
        mentions.append(
            (length, length + len(surface), surface, m.group(2).strip(), m.group(3).strip())
        )
        clean.append(surface)
        length += len(surface)
        pos = m.end()
    clean.append(raw_text[pos:])
    return "".join(clean), mentions


def build_gen_prompt(text, start, end, ctx):
    a = max(0, start - ctx)
    b = min(len(text), end + ctx)
    return GEN_TEMPLATE.format(
        context_before=text[a:start],
        mention=text[start:end],
        context_after=text[end:b],
    )


def build_rerank_prompt(text, start, end, ctx, cands):
    a = max(0, start - ctx)
    b = min(len(text), end + ctx)
    return RERANK_TEMPLATE.format(
        context_before=text[a:start],
        mention=text[start:end],
        context_after=text[end:b],
        candidates=format_candidates(cands),
    )


def extract_prediction(generated_text):
    """First Q-ID in the output, else NONE if stated, else None (unparseable)."""
    m = QID_RE.search(generated_text)
    if m:
        return m.group(0)
    if NONE_TOKEN.lower() in generated_text.lower():
        return NONE_TOKEN
    return None


def main():
    args = parse_args()
    ctx = args.context_chars or (512 if args.mode == "generative" else 384)

    if args.mode == "rerank" and not (
        args.alias_pickle and args.descriptions_jsonl and args.wikimapper_db
    ):
        raise SystemExit(
            "--alias_pickle, --descriptions_jsonl and --wikimapper_db "
            "are required in rerank mode"
        )

    elgold = json.loads(args.elgold_json.read_text(encoding="utf-8"))
    title_to_qid = {}
    for info in elgold.values():
        title_to_qid.update(info["title_to_qid"])

    train_qids = None
    if args.train_jsonl:
        train_qids = set()
        with args.train_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    qid = json.loads(line)["ground_truth_id"]
                    if qid != NONE_TOKEN:
                        train_qids.add(qid)
        print(f"Train Q-ID set: {len(train_qids)} entities")

    alias_table = None
    descriptions = {}
    qid_cache = None
    if args.mode == "rerank":
        print("Loading alias table + descriptions + wikimapper...")
        with args.alias_pickle.open("rb") as f:
            alias_table = pickle.load(f)
        descriptions = load_descriptions(args.descriptions_jsonl)
        # Same resolver as training: candidate lists must match that distribution.
        from wikimapper import WikiMapper
        qid_cache = TitleQidCache(WikiMapper(str(args.wikimapper_db)))

    # ---- Collect evaluation instances -------------------------------------
    instances = []  # dicts: domain, file, surface, gold_qid, prompt, (cand_qids)
    n_nil = 0
    n_unmapped = 0
    for path in sorted(args.elgold_dir.glob("*.txt")):
        domain = category_from_filename(path.name)
        clean, mentions = parse_elgold_doc(path.read_text(encoding="utf-8"))
        for start, end, surface, ne_class, target in mentions:
            if not target:
                n_nil += 1
                continue
            gold_qid = title_to_qid.get(target)
            if gold_qid is None:
                n_unmapped += 1
                continue
            inst = {
                "domain": domain,
                "file": path.name,
                "mention": surface,
                "gold_qid": gold_qid,
            }
            if args.mode == "generative":
                inst["prompt"] = build_gen_prompt(clean, start, end, ctx)
            else:
                cands = build_candidates(
                    alias_table, surface, args.k, qid_cache, descriptions
                )
                inst["prompt"] = build_rerank_prompt(clean, start, end, ctx, cands)
                inst["cand_qids"] = [c["qid"] for c in cands]
            instances.append(inst)
            if args.limit and len(instances) >= args.limit:
                break
        if args.limit and len(instances) >= args.limit:
            break

    print(f"Instances: {len(instances)} (skipped: {n_nil} NIL, {n_unmapped} unmapped)")

    # ---- Load model ---------------------------------------------------------
    print(f"Loading adapter from {args.adapter} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Batched greedy inference ------------------------------------------
    for i in range(0, len(instances), args.batch_size):
        batch = instances[i:i + args.batch_size]
        chats = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": inst["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for inst in batch
        ]
        inputs = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        for inst, seq in zip(batch, new_tokens):
            text = tokenizer.decode(seq, skip_special_tokens=True)
            inst["raw_output"] = text.strip()
            inst["pred"] = extract_prediction(text)
        done = min(i + args.batch_size, len(instances))
        print(f"  {done}/{len(instances)}", end="\r")
    print()

    # ---- Metrics ------------------------------------------------------------
    per_domain = defaultdict(lambda: defaultdict(int))
    for inst in instances:
        d = inst["domain"]
        stats = per_domain[d]
        stats["n"] += 1
        correct = inst["pred"] == inst["gold_qid"]
        stats["correct"] += int(correct)
        if inst["pred"] is None:
            stats["unparseable"] += 1
        if inst["pred"] == NONE_TOKEN:
            stats["pred_none"] += 1
        if args.mode == "rerank":
            stats["cand_recall"] += int(inst["gold_qid"] in inst.get("cand_qids", []))
        if train_qids is not None:
            seen = inst["gold_qid"] in train_qids
            stats["n_seen" if seen else "n_unseen"] += 1
            if correct:
                stats["correct_seen" if seen else "correct_unseen"] += 1

    report = {"mode": args.mode, "adapter": str(args.adapter), "domains": {}}
    total_n = total_c = 0
    print(f"\n{'Domain':30s} {'n':>5s} {'acc@1':>7s}", end="")
    if args.mode == "rerank":
        print(f" {'recall@k':>9s} {'NONE%':>6s}", end="")
    if train_qids is not None:
        print(f" {'acc seen':>9s} {'acc unseen':>10s}", end="")
    print()
    for d, s in sorted(per_domain.items()):
        acc = s["correct"] / s["n"]
        total_n += s["n"]; total_c += s["correct"]
        entry = {"n": s["n"], "accuracy": round(acc, 4),
                 "unparseable": s["unparseable"]}
        print(f"{d:30s} {s['n']:5d} {100 * acc:6.1f}%", end="")
        if args.mode == "rerank":
            entry["candidate_recall"] = round(s["cand_recall"] / s["n"], 4)
            entry["pred_none_rate"] = round(s["pred_none"] / s["n"], 4)
            print(f" {100 * entry['candidate_recall']:8.1f}% {100 * entry['pred_none_rate']:5.1f}%", end="")
        if train_qids is not None:
            acc_seen = s["correct_seen"] / s["n_seen"] if s["n_seen"] else 0.0
            acc_unseen = s["correct_unseen"] / s["n_unseen"] if s["n_unseen"] else 0.0
            entry["n_seen"] = s["n_seen"]; entry["accuracy_seen"] = round(acc_seen, 4)
            entry["n_unseen"] = s["n_unseen"]; entry["accuracy_unseen"] = round(acc_unseen, 4)
            print(f" {100 * acc_seen:8.1f}% {100 * acc_unseen:9.1f}%", end="")
        print()
        report["domains"][d] = entry

    report["global"] = {
        "n": total_n,
        "accuracy": round(total_c / total_n, 4) if total_n else None,
        "n_nil_skipped": n_nil,
        "n_unmapped_skipped": n_unmapped,
    }
    print(f"\n{'GLOBAL':30s} {total_n:5d} {100 * total_c / total_n:6.1f}%")

    out_json = Path(f"{args.output_prefix}_report.json")
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    out_csv = Path(f"{args.output_prefix}_predictions.csv")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "file", "mention", "gold_qid", "pred", "raw_output", "correct"])
        for inst in instances:
            w.writerow([
                inst["domain"], inst["file"], inst["mention"], inst["gold_qid"],
                inst["pred"], inst["raw_output"], int(inst["pred"] == inst["gold_qid"]),
            ])
    print(f"Report: {out_json}\nPredictions: {out_csv}")


if __name__ == "__main__":
    main()
