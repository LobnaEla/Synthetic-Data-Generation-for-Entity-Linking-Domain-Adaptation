import argparse
import csv
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


def run_setup(wikimapper_dir: Path):
    wikimapper_dir.mkdir(parents=True, exist_ok=True)
    db_path = wikimapper_dir / "index_enwiki-latest.db"

    if db_path.exists():
        print(f"[SETUP] DB already exists: {db_path}")
        print("[SETUP] Delete it manually to rebuild.")
        return db_path

    print(f"[SETUP] Downloading Wikipedia dump into {wikimapper_dir} ...")
    subprocess.run(
        ["wikimapper", "download", "enwiki-latest", "--dir", str(wikimapper_dir)],
        check=True,
    )

    print(f"[SETUP] Building SQLite DB ...")
    subprocess.run(
        ["wikimapper", "create", "enwiki-latest",
         "--dumpdir", str(wikimapper_dir),
         "--target", str(db_path)],
        check=True,
    )

    print(f"[SETUP] Done: {db_path}")
    return db_path


# Regex detecting "glued" words: a lowercase letter immediately followed by an
# uppercase letter with no space, e.g. "AlbaniaPeople", "AfghanistanCommunications".
# Canonical Wikipedia titles never look like this.
_GLUED_RE = re.compile(r"[a-z][A-Z]")


def _canonicality_score(title: str) -> tuple:
    """
    Return a sort key: lower is better (more canonical).

    Priority (lexicographic tuple):
      0. Has glued CamelCase words?          0 = no (good), 1 = yes (bad redirect)
      1. Contains a disambiguation suffix?   0 = no, 1 = yes  (neutral, keep if needed)
      2. Title length                         shorter is slightly preferred as tie-break
    """
    has_glued   = int(bool(_GLUED_RE.search(title)))
    has_disam   = int(bool(re.search(r"\(.*\)$", title)))
    return (has_glued, has_disam, len(title))


def main():
    parser = argparse.ArgumentParser(
        description="Extract all Wikipedia titles + QIDs from wikimapper DB."
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Download dump + build wikimapper DB (one time).",
    )
    parser.add_argument(
        "--wikimapper-dir", type=Path, default=Path("data/wikimapper"),
        help="Directory for wikimapper dump + DB (default: data/wikimapper).",
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help="Path to wikimapper SQLite DB. If --setup, auto-detected.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/wikidata_qid_titles.csv"),
        help="Output CSV path (default: data/wikidata_qid_titles.csv)",
    )
    args = parser.parse_args()

    # ── Setup mode ──────────────────────────────────────────────────────────
    if args.setup:
        db_path = run_setup(args.wikimapper_dir)
        if args.db is None:
            args.db = db_path

    if args.db is None:
        args.db = args.wikimapper_dir / "index_enwiki-latest.db"

    if not args.db.exists():
        print(f"[ERROR] DB not found: {args.db}")
        print("Run with --setup first:")
        print(f"  python {sys.argv[0]} --setup --wikimapper-dir data/wikimapper")
        return

    # ── Read ─────────────────────────────────────────────────────────────────
    print(f"Reading from {args.db} ...")
    t0 = time.time()

    conn = sqlite3.connect(str(args.db))
    rows = conn.execute(
        "SELECT wikidata_id, wikipedia_title FROM mapping "
        "WHERE wikidata_id IS NOT NULL AND wikipedia_title IS NOT NULL"
    ).fetchall()
    conn.close()

    print(f"  {len(rows):,} rows read (with redirects) in {time.time() - t0:.1f}s")

    # ── Deduplicate: keep the most canonical title per QID ───────────────────
    #
    # The wikimapper DB contains both real article titles and redirect titles
    # for the same Wikidata QID. Redirect titles are often concatenated strings
    # without spaces (e.g. "AlbaniaHistory", "AfghanistanCommunications").
    #
    # Strategy: for each QID collect all candidate titles, then pick the one
    # with the best canonicality score (see _canonicality_score).
    #
    qid_to_candidates: dict[str, list[str]] = {}
    for qid, title in rows:
        title_clean = title.replace("_", " ")
        qid_to_candidates.setdefault(qid, []).append(title_clean)

    qid_to_title = {
        qid: min(titles, key=_canonicality_score)
        for qid, titles in qid_to_candidates.items()
    }
    rows = [(qid, title) for qid, title in qid_to_title.items()]

    print(f"  {len(rows):,} unique entities after deduplication")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "title"])
        for qid, title in rows:
            writer.writerow([qid, title])

    print(f"  Written: {args.out} ({len(rows):,} entities)")
    print(f"  Total time: {time.time() - t0:.1f}s")

    # ── Sanity check ──────────────────────────────────────────────────────────
    print("\nSanity check (first 5 rows):")
    with open(args.out, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if i >= 5:
                break
            print(f"  {row[0]:>12s}  {row[1]}")


if __name__ == "__main__":
    main()