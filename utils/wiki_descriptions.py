import json
import time
from pathlib import Path

import requests

CACHE_PATH = Path("data/wiki_descriptions_cache.json")
API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
HEADERS = {"User-Agent": "EntityLinkingProject/1.0"}


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def fetch_description(title: str, cache: dict | None = None) -> str:
    if cache is None:
        cache = _load_cache()

    if title in cache:
        return cache[title]

    try:
        url = API_URL.format(title=title.replace(" ", "_"))
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            desc = data.get("description", "") or ""
            # Use extract_html's first sentence if description is too short
            if len(desc) < 10:
                desc = data.get("extract", "").split(". ")[0]
            cache[title] = desc
            return desc
        else:
            cache[title] = ""
            return ""
    except Exception:
        cache[title] = ""
        return ""


def fetch_descriptions_batch(titles: list[str]) -> list[str]:
    cache = _load_cache()
    results = []
    fetched = 0

    for title in titles:
        desc = fetch_description(title, cache)
        results.append(desc)
        if title not in cache:
            fetched += 1
            if fetched % 100 == 0:
                _save_cache(cache)
                time.sleep(0.1)  # rate limiting

    _save_cache(cache)
    return results