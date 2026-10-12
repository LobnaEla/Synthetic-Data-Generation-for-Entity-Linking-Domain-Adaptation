import csv
from functools import lru_cache

DEFAULT_TITLES_PATH = "data/wikidata_qid_titles.csv"


@lru_cache(maxsize=4)
def load_title_to_qid(titles_path: str = DEFAULT_TITLES_PATH) -> dict[str, str]:
    """Title -> Wikidata QID lookup, built once and cached per path."""
    mapping = {}
    with open(titles_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("title") or "").strip()
            qid = (row.get("qid") or "").strip()
            if title and qid:
                mapping[title] = qid
    return mapping
