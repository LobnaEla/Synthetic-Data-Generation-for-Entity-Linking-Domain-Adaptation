"""Create versioned train/val splits from a filtered ZELDA JSONL.

The benchmark test set is ELGOLD itself, so no ZELDA test split is held out:
this produces train + val only. Val is used to monitor loss during training.

The split is deterministic (fixed seed, stable shuffle) and a manifest.json is
written next to the splits with counts, per-domain distribution, the seed, and
sha256 hashes of the output files — so branch (3) can verify it trains against
the exact same protocol.

Usage:
    python make_splits.py zelda_filtered_train.jsonl splits/generative --val_ratio 0.01
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Versioned train/val split for filtered ZELDA data.")
    p.add_argument("input_jsonl", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--val_ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def domain_counts(lines):
    c = Counter()
    for line in lines:
        d = json.loads(line).get("domain", "unknown")
        for dom in (d if isinstance(d, list) else [d]):
            c[dom] += 1
    return dict(c.most_common())


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.input_jsonl.open("r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    rng = random.Random(args.seed)
    rng.shuffle(lines)

    n_val = max(1, int(len(lines) * args.val_ratio))
    val_lines, train_lines = lines[:n_val], lines[n_val:]

    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "val.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with val_path.open("w", encoding="utf-8") as f:
        f.writelines(val_lines)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(args.input_jsonl),
        "source_sha256": sha256_of(args.input_jsonl),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "n_total": len(lines),
        "n_train": len(train_lines),
        "n_val": len(val_lines),
        "train_sha256": sha256_of(train_path),
        "val_sha256": sha256_of(val_path),
        "train_domain_counts": domain_counts(train_lines),
        "val_domain_counts": domain_counts(val_lines),
        "note": "Test set is the ELGOLD benchmark itself; no ZELDA test split is held out.",
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Total examples: {len(lines)}")
    print(f"  train: {len(train_lines)}  -> {train_path}")
    print(f"  val:   {len(val_lines)}  -> {val_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
