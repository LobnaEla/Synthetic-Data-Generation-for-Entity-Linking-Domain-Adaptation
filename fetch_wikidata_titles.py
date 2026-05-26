import argparse
import time
from pathlib import Path

import requests


SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

HEADERS = {
    "User-Agent": "EntityLinkingProject/1.0",
    "Accept": "application/sparql-results+json",
}

BATCH_SIZE = 500
SLEEP_OK = 0.5
SLEEP_RETRY = 20
MAX_RETRIES = 5


QUERY_TEMPLATE = """
SELECT ?item ?title WHERE {{
  VALUES ?item {{ {qid_values} }}

  ?sitelink schema:about ?item ;
            schema:isPartOf <https://en.wikipedia.org/> ;
            schema:name ?title .
}}
"""


def build_query(qid_min: int, qid_max: int) -> str:
    qid_values = " ".join(f"wd:Q{i}" for i in range(qid_min, qid_max))
    return QUERY_TEMPLATE.format(qid_values=qid_values)


def fetch_batch(qid_min: int, qid_max: int):
    query = build_query(qid_min, qid_max)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=HEADERS,
                timeout=60,
            )

            if resp.status_code == 200:
                data = resp.json()
                rows = []

                for b in data["results"]["bindings"]:
                    qid = b["item"]["value"].split("/")[-1]
                    title = b["title"]["value"]
                    rows.append((qid, title))

                return rows

            elif resp.status_code == 429:
                wait = SLEEP_RETRY * attempt
                print(f"    [429 Rate limit] waiting {wait}s...")
                time.sleep(wait)

            else:
                print(f"    [HTTP {resp.status_code}] {resp.text[:300]}")
                time.sleep(SLEEP_RETRY)

        except requests.exceptions.RequestException as e:
            print(f"    [Request error] {e}")
            time.sleep(SLEEP_RETRY)

        except ValueError as e:
            print(f"    [JSON error] {e}")
            time.sleep(SLEEP_RETRY)

    print(f"    [SKIP] Q{qid_min} → Q{qid_max}")
    return None


def load_progress(progress_file: Path) -> int:
    if progress_file.exists():
        try:
            return int(progress_file.read_text().strip())
        except ValueError:
            return 1
    return 1


def save_progress(progress_file: Path, qid: int) -> None:
    progress_file.write_text(str(qid))


def fetch_all(out_path: Path, max_qid: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_file = out_path.with_suffix(".progress")

    start_qid = load_progress(progress_file)

    mode = "a" if out_path.exists() and start_qid > 1 else "w"

    total = 0

    with open(out_path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("qid,title\n")

        qid_min = start_qid

        while qid_min <= max_qid:
            qid_max = min(qid_min + BATCH_SIZE, max_qid + 1)

            print(f"Fetching Q{qid_min:,} → Q{qid_max - 1:,} ...", end=" ", flush=True)

            rows = fetch_batch(qid_min, qid_max)

            if rows is not None:
                for qid, title in rows:
                    safe_title = title.replace('"', '""')
                    f.write(f'{qid},"{safe_title}"\n')

                f.flush()
                total += len(rows)

                print(f"{len(rows):,} rows")
                save_progress(progress_file, qid_max)
                time.sleep(SLEEP_OK)

            else:
                print("SKIPPED")
                save_progress(progress_file, qid_max)

            qid_min = qid_max

    print(f"Done. Saved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="../data/wikidata_qid_titles.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-qid",
        type=int,
        default=2_000_000,
        help="Upper QID bound",
    )

    args = parser.parse_args()
    fetch_all(Path(args.out), args.max_qid)