import argparse
import csv
import glob
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Auto-patch wikimapper's processor.py to disable redirect propagation.
# Without this patch, redirect pages get a wikidata_id copied from the
# canonical page, making it impossible to distinguish them at extraction.
# With the patch, only canonical pages retain a wikidata_id.
# ---------------------------------------------------------------------------

_REDIRECT_BLOCK_START = '_logger.info("Parsing redirects dump")'
_REDIRECT_BLOCK_END = "conn.commit()"
_PATCH_MARKER = "# PATCHED: redirect section disabled"

# If the DB has more than this many entities with a wikidata_id,
# it likely includes redirects (canonical pages are ~6-7M).
_MAX_EXPECTED_CANONICAL = 10_000_000


def _find_processor() -> Path | None:
    """Locate wikimapper's processor.py inside the current venv."""
    matches = glob.glob(".venv/**/wikimapper/processor.py", recursive=True)
    return Path(matches[0]) if matches else None


def _patch_processor(proc_path: Path) -> bool:
    """Comment out the redirect-parsing section. Returns True if patched."""
    text = proc_path.read_text(encoding="utf-8")

    if _PATCH_MARKER in text:
        print("[PATCH] processor.py already patched.")
        return False

    if _REDIRECT_BLOCK_START not in text:
        print("[PATCH] WARNING: could not find redirect section — skipping patch.")
        return False

    lines = text.splitlines(keepends=True)
    new_lines = []
    in_block = False
    patched = False

    for line in lines:
        if not in_block and _REDIRECT_BLOCK_START in line:
            new_lines.append(
                f"    {_PATCH_MARKER} by extract_all_entities.py\n"
            )
            in_block = True

        if in_block:
            new_lines.append("    # " + line.lstrip() if line.strip() else line)
            if _REDIRECT_BLOCK_END in line:
                in_block = False
                patched = True
        else:
            new_lines.append(line)

    if patched:
        proc_path.write_text("".join(new_lines), encoding="utf-8")
        print(f"[PATCH] Redirect section commented out in {proc_path}")
    else:
        print("[PATCH] WARNING: could not locate end of redirect section.")

    return patched


# ---------------------------------------------------------------------------
# Setup: download dump + build DB
# ---------------------------------------------------------------------------

def run_setup(wikimapper_dir: Path):
    wikimapper_dir.mkdir(parents=True, exist_ok=True)
    db_path = wikimapper_dir / "index_enwiki-latest.db"

    if db_path.exists():
        print(f"[SETUP] DB already exists: {db_path}")
        print("[SETUP] Delete it manually to rebuild.")
        return db_path

    # Auto-patch before building the DB
    proc_path = _find_processor()
    if proc_path is None:
        print("[SETUP] ERROR: wikimapper not installed (processor.py not found).")
        print("        Run: pip install wikimapper")
        sys.exit(1)
    _patch_processor(proc_path)

    print(f"[SETUP] Downloading Wikipedia dump into {wikimapper_dir} ...")
    subprocess.run(
        ["wikimapper", "download", "enwiki-latest", "--dir", str(wikimapper_dir)],
        check=True,
    )

    print("[SETUP] Building SQLite DB ...")
    subprocess.run(
        ["wikimapper", "create", "enwiki-latest",
         "--dumpdir", str(wikimapper_dir),
         "--target", str(db_path)],
        check=True,
    )

    print(f"[SETUP] Done: {db_path}")
    return db_path


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

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
        help="Path to wikimapper SQLite DB.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/wikidata_qid_titles.csv"),
        help="Output CSV path (default: data/wikidata_qid_titles.csv)",
    )
    args = parser.parse_args()

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

    print(f"Reading from {args.db} ...")
    t0 = time.time()

    conn = sqlite3.connect(str(args.db))

    rows = conn.execute(
        """
        SELECT wikidata_id, wikipedia_title
        FROM mapping
        WHERE wikidata_id IS NOT NULL
          AND wikipedia_title IS NOT NULL
        """
    ).fetchall()
    conn.close()

    # Sanity check: if too many rows, the DB likely includes redirects
    if len(rows) > _MAX_EXPECTED_CANONICAL:
        print(f"  [WARN] {len(rows):,} entities found — expected ~6-7M for canonical only.")
        print("         The DB may have been built without the wikimapper patch.")
        print("         Delete the .db file and re-run with --setup to rebuild.")

    rows = [(qid, title.replace("_", " ")) for qid, title in rows]
    print(f"  {len(rows):,} entities extracted in {time.time() - t0:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "title"])
        writer.writerows(rows)

    print(f"  Written: {args.out} ({len(rows):,} entities)")
    print(f"  Total time: {time.time() - t0:.1f}s")

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